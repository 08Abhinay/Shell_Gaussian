#!/usr/bin/env python3
"""Build and audit the experimental 11-D canonical quadratic scalar field."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import traceback
from dataclasses import fields

if __name__ == "__main__":
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"

import numpy as np

from foot_prior.anatomical_volume import load_canonical_anatomical_volume
from foot_prior.anatomical_fibers import (
    CanonicalSemanticField, _layout, _refine_selected_cells, audit_semantic_field,
    evaluate_semantic_field, refine_scalar_neighborhoods,
    semantic_configuration, solve_canonical_semantic_field,
    load_canonical_semantic_field,
    build_anatomical_fiber_field, load_anatomical_fiber_field, fiber_configuration,
    instance_to_semantic, semantic_to_instance, canonical_to_semantic,
    sample_fiber_surface, fiber_category_codes, fiber_category_name,
    summarize_fiber_coverage, selected_fiber_paths, FiberStatus, FiberQueryResult, FiberCoordinates,
)
from foot_prior.anatomy import array_digest
from scripts.run_anatomical_volume import _write_deterministic_npz

ARTIFACTS = ("semantic_field.json", "semantic_field.npz", "semantic_field.vtk")
_FIBER_WORKER_STATE = None


def _json_atomic(path, payload):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile('w',dir=path.parent,prefix='.json-',delete=False) as stream:
        temporary=Path(stream.name)
        json.dump(payload,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    os.replace(temporary,path)


def _write_paths(path, curves, reasons):
    """Small VTK XML polyline visualization; no geometry libraries required."""
    points=np.concatenate(curves) if curves else np.empty((0,3))
    offsets=np.cumsum([len(x) for x in curves]); connect=np.arange(len(points))
    def numbers(a): return ' '.join(str(x) for x in np.asarray(a).ravel())
    text=('<?xml version="1.0"?>\n<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">'
          f'<PolyData><Piece NumberOfPoints="{len(points)}" NumberOfLines="{len(curves)}">'
          '<Points><DataArray type="Float64" NumberOfComponents="3" format="ascii">'
          +numbers(points)+'</DataArray></Points><Lines><DataArray type="Int64" Name="connectivity" format="ascii">'
          +numbers(connect)+'</DataArray><DataArray type="Int64" Name="offsets" format="ascii">'
          +numbers(offsets)+'</DataArray></Lines><CellData><DataArray type="Int32" Name="stop_reason" format="ascii">'
          +numbers(reasons)+'</DataArray></CellData></Piece></PolyData></VTKFile>\n')
    path.write_text(text)


def _write_fiber_field(directory, field, inputs):
    directory.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory,prefix='.fiber-') as temporary:
        staging=Path(temporary)
        _write_deterministic_npz(staging/'fiber_field.npz',directions=field.directions,exclusions=field.exclusions)
        c=field.canonical
        with (staging/'fiber_field.vtk').open('w') as stream:
            stream.write('# vtk DataFile Version 3.0\nExperimental directions, NOT full-domain fibers\nASCII\nDATASET UNSTRUCTURED_GRID\n')
            stream.write(f'POINTS {len(c.volume_vertices)} double\n');np.savetxt(stream,c.volume_vertices,fmt='%.17g')
            stream.write(f'CELLS {len(c.tetrahedra)} {5*len(c.tetrahedra)}\n')
            np.savetxt(stream,np.c_[np.full(len(c.tetrahedra),4),c.tetrahedra],fmt='%d')
            stream.write(f'CELL_TYPES {len(c.tetrahedra)}\n');np.savetxt(stream,np.full(len(c.tetrahedra),10),fmt='%d')
            stream.write(f'POINT_DATA {len(c.volume_vertices)}\nVECTORS direction double\n')
            np.savetxt(stream,field.directions,fmt='%.17g')
            stream.write(f'CELL_DATA {len(c.tetrahedra)}\nSCALARS exclusion int 1\nLOOKUP_TABLE default\n')
            np.savetxt(stream,field.exclusions,fmt='%d')
        report=dict(schema_version=1,stage='canonical_anatomical_fiber_field',status='coverage_review_required',
            configuration=fiber_configuration(),inputs=inputs,source_geometry_digest=field.scalar.source_geometry_digest,
            scalar_digest=array_digest(field.scalar.coefficients,field.scalar.unique_edges),
            field_digest=array_digest(field.directions,field.exclusions),diagnostics=field.diagnostics,
            region_names={k:v['names'] for k,v in field.region_labels.items()},fibers_globally_validated=False)
        _json_atomic(staging/'fiber_field.json',report)
        load_anatomical_fiber_field(c,field.scalar,staging,field.region_labels)
        for name in ['fiber_field.npz','fiber_field.vtk','fiber_field.json']:
            os.replace(staging/name,directory/name)


def _join_results(results):
    def cat(name): return np.concatenate([getattr(r,name) for r in results])
    coord=FiberCoordinates(*(np.concatenate([getattr(r.coordinates,n) for r in results])
                              for n in ['face_indices','barycentric_weights','semantic_r']))
    arrays=[cat(f.name) for f in fields(FiberQueryResult)[1:-2]]
    labels={k:np.concatenate([r.label_weights[k] for r in results]) for k in results[0].label_weights}
    provenance={k:np.concatenate([r.correspondence[k] for r in results]) for k in results[0].correspondence}
    return FiberQueryResult(coord,*arrays,labels,provenance)


def _query_chunks(field, instance, points, name, phase, tolerance_scale=1.):
    results=[]; start=time.monotonic()
    for offset in range(0,len(points),256):
        result=instance_to_semantic(field,instance,points[offset:offset+256],input_frame='normalized_shoe',
                                   tolerance_scale=tolerance_scale)
        results.append(result)
        elapsed=time.monotonic()-start; complete=min(offset+256,len(points))
        print(f'{name} {phase}: {complete}/{len(points)} queries; '
              f'anatomical={sum(int(r.footwear_support_mask.sum()) for r in results)}; '
              f'{complete/max(elapsed,1e-9):.2f} queries/s; elapsed={elapsed:.1f}s',flush=True)
    return _join_results(results)


def _target_faces(mesh, sample_faces, codes, instance, field):
    bad=np.unique(sample_faces[codes != FiberStatus.VALID_ANATOMICAL])
    edge_pairs=mesh.faces[:,[[0,1],[0,2],[1,2]]]
    if len(bad):
        edges=set(map(tuple,np.sort(edge_pairs[bad].reshape(-1,2),axis=1)))
        neighbors=[k for k,ee in enumerate(edge_pairs) if any(tuple(sorted(e)) in edges for e in ee)]
    else:
        neighbors=[]
    normalized_triangles=mesh.vertices[mesh.faces]
    lower,upper=normalized_triangles.min(1),normalized_triangles.max(1)
    seeds=np.asarray(field.diagnostics['original_saddle_cells'],dtype=int)
    cells=instance.instance_vertices[field.canonical.tetrahedra[seeds]]
    padding=field.diagnostics['normalized_boundary_edge_median']*field.diagonal
    nearby=np.zeros(len(mesh.faces),dtype=bool)
    for box in cells:
        nearby |= np.all(upper >= box.min(0)-padding,axis=1)&np.all(lower <= box.max(0)+padding,axis=1)
    return np.unique(np.r_[bad,np.asarray(neighbors,dtype=int),np.flatnonzero(nearby)])


def _save_coverage_overlay(path, mesh, face_ids, codes):
    """Face colors describe sampled evidence only. Unobserved faces stay grey."""
    import trimesh
    colors=np.tile(np.array([170,175,180,255],dtype=np.uint8),(len(mesh.faces),1))
    evidence={}
    for face,code in zip(face_ids,codes):
        evidence.setdefault(int(face),set()).add(int(code))
    for face,values in evidence.items():
        if len(values)>1: color=[235,173,52,255]
        elif values=={int(FiberStatus.VALID_ANATOMICAL)}:color=[55,171,111,255]
        elif values=={int(FiberStatus.VALID_ARTIFICIAL_CAP)}:color=[143,97,185,255]
        elif min(values)>=1000:color=[73,120,184,255]
        else:color=[207,62,68,255]
        colors[face]=color
    output=trimesh.Trimesh(vertices=mesh.vertices,faces=mesh.faces,process=False,validate=False)
    output.visual.face_colors=colors;output.export(path)


def _coverage_picture(directory, mesh, sample_faces, sample_weights, codes, name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    points=np.einsum('ni,nij->nj',sample_weights,mesh.vertices[mesh.faces[sample_faces]])
    figures,axes=plt.subplots(1,2,figsize=(13,6))
    palette=lambda k: '#37ab6f' if k==0 else '#8f61b9' if k==1 else '#4978b8' if k>=1000 else '#cf3e44'
    # Limit background rendering only; all sampled locations are retained.
    indices=np.linspace(0,len(mesh.faces)-1,min(len(mesh.faces),40000),dtype=int)
    triangles=mesh.vertices[mesh.faces[indices]]
    for ax,uv,title in zip(axes,[(0,1),(0,2)],['Side projection','Across-foot projection']):
        ax.add_collection(PolyCollection(triangles[:,:,uv],facecolor='#c7ccd1',edgecolor='none',alpha=.16))
        order=np.argsort(codes==0)[::-1]
        ax.scatter(points[order,uv[0]],points[order,uv[1]],c=[palette(k) for k in codes[order]],s=3,alpha=.8)
        ax.autoscale_view();ax.set_aspect('equal');ax.set_title(title)
    figures.suptitle(f'{name}: observed fiber coverage (sampled, not exact area)')
    figures.text(.5,.025,'Green: anatomical  |  Purple: artificial cap  |  Red: fiber unavailable  |  Blue: invalid in 11-C\nGrey background: shoe geometry; unsampled areas are not certified.',ha='center',fontsize=10)
    figures.tight_layout(rect=(0,.08,1,.94));figures.savefig(directory/'coverage.png',dpi=160);plt.close(figures)


def _canonical_points_to_instance(field, instance, points):
    values=np.asarray(points,dtype=float)
    output=np.full(values.shape,np.nan)
    finite=np.isfinite(values).all(axis=1)
    if finite.any():
        located=field.locator.locate(values[finite])
        valid=np.flatnonzero(finite)[located.mappable_mask]
        if len(valid):
            output[valid]=instance.canonical_to_instance(
                located.coordinates.tetrahedron_indices[located.mappable_mask],
                located.coordinates.barycentric_weights[located.mappable_mask],
                output_frame='normalized_shoe')
    return output


def _alignment_picture(path, mesh, instance, field, result, curves, name):
    """Show the shoe, fitted computational anatomy, and sampled fiber geometry."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    canonical=field.canonical
    inner_vertices=instance.instance_vertices[canonical.computational_inner_vertex_indices]
    inner_triangles=inner_vertices[canonical.computational_inner_faces]
    shoe_ids=np.linspace(0,len(mesh.faces)-1,min(len(mesh.faces),30000),dtype=int)
    inner_ids=np.linspace(0,len(inner_triangles)-1,min(len(inner_triangles),20000),dtype=int)
    valid=np.flatnonzero(result.mappable_mask)
    if len(valid):
        selected=valid[np.linspace(0,len(valid)-1,min(len(valid),128),dtype=int)]
        origins=_canonical_points_to_instance(field,instance,result.inner_origins[selected])
        endpoints=_canonical_points_to_instance(field,instance,result.outer_endpoints[selected])
    else:
        origins=np.empty((0,3));endpoints=np.empty((0,3))

    figure,axes=plt.subplots(1,2,figsize=(14,6))
    for ax,uv,title in zip(axes,[(0,1),(0,2)],['Side projection','Across-foot projection']):
        ax.add_collection(PolyCollection(mesh.vertices[mesh.faces[shoe_ids]][:,:,uv],
            facecolor='#6baed6',edgecolor='none',alpha=.18))
        ax.add_collection(PolyCollection(inner_triangles[inner_ids][:,:,uv],
            facecolor='#f28e2b',edgecolor='none',alpha=.22))
        for curve in curves:
            ax.plot(curve[:,uv[0]],curve[:,uv[1]],color='#7b3294',linewidth=.9,alpha=.8)
        if len(origins):
            finite=np.isfinite(origins).all(1);ax.scatter(origins[finite,uv[0]],origins[finite,uv[1]],
                s=9,c='#1a9850',label='inner origins',zorder=3)
            finite=np.isfinite(endpoints).all(1);ax.scatter(endpoints[finite,uv[0]],endpoints[finite,uv[1]],
                s=9,c='#d73027',label='outer endpoints',zorder=3)
        ax.autoscale_view();ax.set_aspect('equal');ax.set_title(title)
    axes[0].legend(handles=[
        plt.Line2D([],[],color='#6baed6',linewidth=7,alpha=.5,label='normalized shoe'),
        plt.Line2D([],[],color='#f28e2b',linewidth=7,alpha=.5,label='B3 computational anatomy'),
        plt.Line2D([],[],color='#7b3294',label='selected fibers'),
        plt.Line2D([],[],marker='o',linestyle='',color='#1a9850',label='inner origins'),
        plt.Line2D([],[],marker='o',linestyle='',color='#d73027',label='outer endpoints')],
        loc='best',fontsize=8)
    figure.suptitle(f'{name}: normalized-shoe alignment and experimental fibers')
    figure.tight_layout(rect=(0,0,1,.95));figure.savefig(path,dpi=170);plt.close(figure)


def _audit_shoe(field, instance, mesh, directory, inputs):
    original=instance.convert_points(mesh.vertices,input_frame='normalized_shoe',output_frame='original_shoe')
    name=instance.shoe_name; all_results=[]; histories=[]; previous=0; stable=False
    for count in [4096,16384,65536]:
        face_ids,weights,points,area=sample_fiber_surface(mesh.vertices,mesh.faces,original,count)
        result=_query_chunks(field,instance,points[previous:],name,f'global-{count}')
        all_results.append(result); combined=_join_results(all_results);codes=fiber_category_codes(combined)
        fractions=summarize_fiber_coverage(codes,area)
        histories.append(dict(count=count,categories=fractions))
        if previous:
            old=histories[-2]['categories'];keys=set(old)|set(fractions)
            delta=max(abs(fractions.get(k,{}).get('total_area_fraction',0)-old.get(k,{}).get('total_area_fraction',0)) for k in keys)
            histories[-1]['maximum_fraction_change']=delta;stable=delta<=.002
            if stable or count==65536:break
        previous=count
    targeted_faces=_target_faces(mesh,face_ids,codes,instance,field)
    targeted=None;target_face_ids=np.empty(0,dtype=int);target_weights=np.empty((0,3));target_codes=np.empty(0,dtype=np.int16)
    if len(targeted_faces):
        target_face_ids,target_weights,target_points,_=sample_fiber_surface(mesh.vertices,mesh.faces,original,4096,targeted_faces)
        targeted=_query_chunks(field,instance,target_points,name,'targeted')
        target_codes=fiber_category_codes(targeted)
    success=np.flatnonzero(combined.footwear_support_mask);failed=np.flatnonzero(~combined.footwear_support_mask)
    def choose(ids,n):return ids[np.linspace(0,len(ids)-1,min(n,len(ids)),dtype=int)] if len(ids) else ids
    sensitivity_ids=np.r_[choose(success,512),choose(failed,128)]
    tight=_query_chunks(field,instance,points[sensitivity_ids],name,'tolerance-check',.1)
    sensitivity_codes=fiber_category_codes(tight)
    valid=tight.mappable_mask & combined.mappable_mask[sensitivity_ids]
    drift=np.linalg.norm(tight.inner_origins[valid]-combined.inner_origins[sensitivity_ids[valid]],axis=1)
    changes=int(np.sum(sensitivity_codes != codes[sensitivity_ids]))
    good=np.flatnonzero(tight.mappable_mask);frame_checks={}
    for frame in ['normalized_shoe','original_shoe']:
        if len(good):
            # Explicit public semantic inverse, not interpolation of a stored point.
            rebuilt=semantic_to_instance(field,instance,tight.coordinates.face_indices[good],
                tight.coordinates.barycentric_weights[good],tight.coordinates.semantic_r[good],output_frame=frame)
            expected=points[sensitivity_ids[good]]
            expected=instance.convert_points(expected,input_frame='normalized_shoe',output_frame=frame)
            # Compare in the internal normalized frame so one physical scale is used.
            differences=np.linalg.norm(
                instance.convert_points(rebuilt,input_frame=frame,output_frame='normalized_shoe')-
                instance.convert_points(expected,input_frame=frame,output_frame='normalized_shoe'),axis=1)
            frame_checks[frame]=dict(count=len(good),maximum_error=float(differences.max()) if np.isfinite(differences).all() else None,
                passed=bool(np.isfinite(differences).all() and np.max(differences)<=5e-6*field.diagonal))
        else:frame_checks[frame]=dict(count=0,passed=None,maximum_error=None)
    report=dict(schema_version=1,stage='anatomical_fiber_coverage',shoe_name=name,status='coverage_review_required',
        sampling_status='sampling_stable' if stable else 'sampling_unresolved',inputs=inputs,
        configuration=dict(**fiber_configuration(),sample_stages=[4096,16384,65536],stability_fraction=.002,
                           targeted_sample_budget=4096,sampling='unscrambled_halton_skip_zero'),
        total_surface_area=area,global_sample_count=len(codes),categories=summarize_fiber_coverage(codes,area),
        sampling_history=histories,targeted_sample_count=len(target_codes),
        targeted_status_counts={fiber_category_name(k):int(n) for k,n in zip(*np.unique(target_codes,return_counts=True))},
        sensitivity=dict(count=len(sensitivity_ids),status_changes=changes,
                         maximum_origin_drift=float(drift.max()) if len(drift) else None,
                         frame_checks=frame_checks),
        limitations=['Area estimates are sampled, not exact or confidence intervals.',
                     'Failures have no accepted anatomical origin; their region remains unknown.',
                     'Targeted samples are excluded from global area estimates.',
                     'Observed saddle encounters do not establish causal loss from saddles alone.'],
        overlay_colors=dict(green='observed anatomical',purple='observed artificial cap',red='observed fiber failure',
                            blue='observed 11-C invalid',orange='mixed observations',grey='unsampled'),
        accepted_anatomical_count=int(combined.footwear_support_mask.sum()))
    report['region_label_estimates']={}
    for key,label in combined.label_weights.items():
        report['region_label_estimates'][key]=dict(names=field.region_labels[key]['names'],
            accepted_area_by_label=(np.nansum(label[combined.footwear_support_mask],axis=0)*area/len(codes)).tolist(),
            unassigned_area=float(area*np.sum(~combined.footwear_support_mask)/len(codes)))
    arrays=dict(source_face_indices=face_ids,source_barycentric=weights,status_codes=codes,
        volume_status_codes=combined.volume_status_codes,backward_reasons=combined.backward_reasons,
        forward_reasons=combined.forward_reasons,origin_face_indices=combined.coordinates.face_indices,
        origin_barycentric=combined.coordinates.barycentric_weights,semantic_r=combined.coordinates.semantic_r,
        inner_origins=combined.inner_origins,outer_endpoints=combined.outer_endpoints,
        round_trip_errors=combined.round_trip_errors,integration_steps=combined.integration_steps,
        targeted_face_indices=target_face_ids,targeted_barycentric=target_weights,targeted_status_codes=target_codes,
        sensitivity_indices=sensitivity_ids,sensitivity_status_codes=sensitivity_codes)
    arrays.update({f'label_weights_{k}':v for k,v in combined.label_weights.items()})
    arrays.update({f'correspondence_{k}':v for k,v in combined.correspondence.items()})
    directory.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory,prefix='.coverage-') as temporary:
        staging=Path(temporary)
        _write_deterministic_npz(staging/'fiber_samples.npz',**arrays)
        _save_coverage_overlay(staging/'coverage_overlay.ply',mesh,np.r_[face_ids,target_face_ids],np.r_[codes,target_codes])
        selected=np.unique(np.r_[choose(success,16),choose(failed,8)])
        loc=instance.instance_to_canonical(points[selected],input_frame='normalized_shoe')
        curves,reasons=selected_fiber_paths(field,loc.canonical_points)
        physical=[];physical_reasons=[]
        for curve,reason in zip(curves,reasons):
            located=field.locator.locate(curve); valid_curve=located.mappable_mask
            if not valid_curve.all():continue
            physical.append(instance.canonical_to_instance(located.coordinates.tetrahedron_indices,
                located.coordinates.barycentric_weights,output_frame='normalized_shoe'))
            physical_reasons.append(reason)
        _write_paths(staging/'selected_fibers.vtp',physical,physical_reasons)
        _coverage_picture(staging,mesh,face_ids,weights,codes,name)
        _alignment_picture(staging/'alignment.png',mesh,instance,field,combined,physical,name)
        _json_atomic(staging/'fiber_coverage.json',report)
        for path in sorted(staging.iterdir(),key=lambda p:p.name=='fiber_coverage.json'):
            os.replace(path,directory/path.name)
    return report


def _initialize_fiber_worker(canonical_root,scalar_root,fiber_root,regions,batch_root,
                             containment_root,shoe_paths,inputs,shoe_inputs,output_root):
    """Load immutable canonical data once in each per-shoe worker."""
    from foot_prior.instance_volume_mapping import load_instance_volume_map
    canonical=load_canonical_anatomical_volume(Path(canonical_root))
    scalar=load_canonical_semantic_field(canonical,Path(scalar_root))
    field=load_anatomical_fiber_field(canonical,scalar,Path(fiber_root),regions)
    global _FIBER_WORKER_STATE
    _FIBER_WORKER_STATE=dict(canonical=canonical,field=field,batch_root=Path(batch_root),
        containment_root=Path(containment_root),shoe_paths={k:Path(v) for k,v in shoe_paths.items()},
        inputs=inputs,shoe_inputs=shoe_inputs,output_root=Path(output_root),loader=load_instance_volume_map)


def _audit_shoe_worker(name):
    """Run one independent shoe audit and return its deterministic report."""
    from foot_prior.mesh import load_triangle_mesh
    state=_FIBER_WORKER_STATE
    if state is None:raise RuntimeError('fiber worker was not initialized')
    sources=state['inputs']+state['shoe_inputs'][name]
    for source in sources:
        if _digest(source['path']) != source['sha256']:
            raise ValueError(f'{name}: input changed during experiment')
    instance=state['loader'](state['canonical'],state['batch_root']/name,state['containment_root']/name)
    mesh=load_triangle_mesh(state['shoe_paths'][name])
    return _audit_shoe(state['field'],instance,mesh,state['output_root']/name,sources)


def _run_fibers(args):
    from foot_prior.instance_volume_mapping import load_instance_volume_map
    from foot_prior.mesh import load_triangle_mesh
    from foot_prior.anatomical_volume import _load_extended_reference
    from scripts.run_instance_volume_mapping import _selected_shoes,_preflight_paths
    required=['scalar_field_root','extended_anatomical_surface_root','instance_volume_batch_root','containment_fit_root']
    if any(getattr(args,key) is None for key in required):
        raise ValueError('fibers workflow requires scalar, extended surface, B3 batch, and containment roots')
    root=args.output_root.resolve()
    if root.exists():
        children=list(root.iterdir())
        allowed=(not children or (len(children)==1 and children[0].name=='logs' and children[0].is_dir()
            and all(path.name=='run.log' for path in children[0].iterdir())))
        if not allowed:
            raise FileExistsError('fiber runs require a fresh output directory; no implicit overwrite or retry')
    names,manifest=_selected_shoes(args.instance_volume_batch_root,args.shoes,args.exclude)
    _preflight_paths(names,args.instance_volume_batch_root,args.containment_fit_root)
    c=load_canonical_anatomical_volume(args.anatomical_volume_root)
    scalar_record=json.loads((args.scalar_field_root/'semantic_field.json').read_text())
    if scalar_record.get('refinement'):
        raise ValueError('refined scalar is excluded from fiber experiment')
    scalar=load_canonical_semantic_field(c,args.scalar_field_root)
    reference=_load_extended_reference(args.extended_anatomical_surface_root)
    if reference.geometry_digest != c.extended_surface_digest:raise ValueError('anatomical label geometry mismatch')
    ext_root=args.extended_anatomical_surface_root/'reference'
    metadata=json.loads((ext_root/'canonical_extended_surface.json').read_text())
    with np.load(ext_root/'canonical_extended_surface.npz',allow_pickle=False) as z:
        regions={k:dict(names=metadata['regions'][k]['names'],face_labels=z[f'{k}_face_labels'].tolist())
                 for k in ['longitudinal','surface','component']}
    shoe_paths={};shoe_inputs={}
    for name in names:
        containment_path=args.containment_fit_root/name/'containment_fit.json'
        record=json.loads(containment_path.read_text());path=Path(record['inputs']['normalized_shoe']).resolve(strict=True)
        if path.parent.name != name:raise ValueError('shoe mesh path/name mismatch')
        # Validate every selected shoe before any experiment output is written.
        checked=load_instance_volume_map(c,args.instance_volume_batch_root/name,args.containment_fit_root/name)
        del checked
        mesh=load_triangle_mesh(path);del mesh
        shoe_paths[name]=path
        sources=[path,containment_path,args.instance_volume_batch_root/name/'instance_volume.json',
                 args.instance_volume_batch_root/name/'instance_volume.npz']
        shoe_inputs[name]=[dict(path=str(p),sha256=_digest(p)) for p in sources]
    sources=[args.anatomical_volume_root/'reference'/f'canonical_volume.{s}' for s in ['json','npz']]
    sources += [args.scalar_field_root/f'semantic_field.{s}' for s in ['json','npz']]
    sources += [ext_root/f'canonical_extended_surface.{s}' for s in ['json','npz']]
    inputs=[dict(path=str(p.resolve()),sha256=_digest(p)) for p in sources]
    # Fresh independent stationary audit: saved records alone cannot hide a new seed.
    audit,diagnostics=audit_semantic_field(c,scalar)
    with np.load(args.scalar_field_root/'semantic_field.npz') as z:
        saved=z['stationary_cell_ids'][z['stationary_locations']=='cell_interior']
    seeds=diagnostics['stationary_cell_ids'][diagnostics['stationary_locations']=='cell_interior']
    if not np.array_equal(saved,seeds) or not np.array_equal(seeds,[46812,49996,50003,50035,53211,53385]):
        raise ValueError('fiber experiment requires the audited six-saddle baseline')
    if not scalar.solver['converged'] or audit['checks']['near_flat_cells']:
        raise ValueError('scalar baseline failed fiber preflight')
    root.mkdir(parents=True,exist_ok=True)
    (root/'logs').mkdir(exist_ok=True)
    _json_atomic(root/'fiber_manifest.json',dict(schema_version=1,stage='anatomical_fiber_batch',shoes=names,
        exclusions=sorted(set(args.exclude)|{'sneaker_vibe'}),inputs=inputs,shoe_inputs=shoe_inputs,
        configuration=dict(**fiber_configuration(),jobs=args.jobs),thread_policy={k:os.environ.get(k) for k in
            ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']}))
    field=build_anatomical_fiber_field(c,scalar,seeds,regions)
    _write_fiber_field(root/'reference',field,inputs)
    summary=dict(schema_version=1,status='coverage_review_required',completed=0,failed=0,results={})
    _json_atomic(root/'fiber_summary.json',summary)
    pilots=[n for n in ['birkenstock_arizona_sandal','sandal_1'] if n in names]
    pilot_success=0;pilot_alignment=True;canonical_curves=[];canonical_reasons=[]
    for name in pilots:
        instance=load_instance_volume_map(c,args.instance_volume_batch_root/name,args.containment_fit_root/name)
        mesh=load_triangle_mesh(shoe_paths[name]);original=instance.convert_points(
            mesh.vertices,input_frame='normalized_shoe',output_frame='original_shoe')
        _,_,points,_=sample_fiber_surface(mesh.vertices,mesh.faces,original,64)
        result=_query_chunks(field,instance,points,name,'pilot-64')
        pilot_success+=int(result.footwear_support_mask.sum())
        names_component=field.region_labels['component']['names']
        anatomical=np.flatnonzero(result.footwear_support_mask)
        skin_indices=[names_component.index(label) for label in ('foot_skin','ankle_transition')]
        non_calf=int(np.sum(np.nansum(result.label_weights['component'][anatomical][:,skin_indices],axis=1)>1e-8))
        aligned=bool(len(anatomical) and non_calf)
        pilot_alignment &= aligned
        _json_atomic(root/'logs'/f'{name}_pilot.json',dict(anatomical_count=int(len(anatomical)),
            non_calf_anatomical_origin_count=non_calf,alignment_passed=aligned,
            statuses={fiber_category_name(k):int(v) for k,v in zip(*np.unique(fiber_category_codes(result),return_counts=True))}))
        selected=result.canonical_points[result.footwear_support_mask][:8]
        curves,reasons=selected_fiber_paths(field,selected);canonical_curves.extend(curves);canonical_reasons.extend(reasons)
        physical=[]
        for curve in curves:
            located=field.locator.locate(curve)
            if located.mappable_mask.all():
                physical.append(instance.canonical_to_instance(located.coordinates.tetrahedron_indices,
                    located.coordinates.barycentric_weights,output_frame='normalized_shoe'))
        _alignment_picture(root/'logs'/f'{name}_pilot_alignment.png',mesh,instance,field,result,physical,name)
    _write_paths(root/'reference'/'selected_fibers.vtp',canonical_curves,canonical_reasons)
    if pilots and (not pilot_success or not pilot_alignment):
        summary.update(status=('pilot_failed_no_complete_anatomical_fibers' if not pilot_success
            else 'pilot_failed_normalized_alignment'))
        _json_atomic(root/'fiber_summary.json',summary)
        print('STOP: pilot fibers or their normalized anatomical alignment failed; full audit not launched.',flush=True)
        return 1
    print(f'Starting {len(names)} shoe audits with {args.jobs} worker(s).',flush=True)
    with ProcessPoolExecutor(max_workers=min(args.jobs,len(names)),initializer=_initialize_fiber_worker,
            initargs=(str(args.anatomical_volume_root),str(args.scalar_field_root),str(root/'reference'),regions,
                      str(args.instance_volume_batch_root),str(args.containment_fit_root),
                      {k:str(v) for k,v in shoe_paths.items()},inputs,shoe_inputs,str(root))) as executor:
        futures={executor.submit(_audit_shoe_worker,name):name for name in names}
        for future in as_completed(futures):
            name=futures[future]
            try:
                report=future.result()
                summary['results'][name]=dict(status=report['status'],categories=report['categories'],
                    sampling_status=report['sampling_status'],sensitivity=report['sensitivity'])
                summary['completed']+=1
            except Exception as error:
                traceback.print_exception(type(error),error,error.__traceback__)
                summary['failed']+=1
                failure=dict(status='audit_failed',error=f'{type(error).__name__}: {error}')
                summary['results'][name]=failure
                _json_atomic(root/name/'fiber_coverage.json',failure)
            _json_atomic(root/'fiber_summary.json',summary)
            print(f'Finished {name}: completed={summary["completed"]}; failed={summary["failed"]}',flush=True)
    print(f'DONE: coverage_review_required; {root}',flush=True)
    return int(summary['failed']>0)


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _load_field(canonical, directory):
    return load_canonical_semantic_field(canonical, directory)


def _write_vtk(path, canonical, field, diagnostics):
    edges = field.unique_edges
    points = np.vstack((canonical.volume_vertices, canonical.volume_vertices[edges].mean(axis=1)))
    # Bernstein order 01,02,03,12,13,23 -> VTK 01,12,20,03,13,23.
    cells = field.cell_coefficients[:, [0, 1, 2, 3, 4, 7, 5, 6, 8, 9]]
    nv = len(canonical.volume_vertices)
    semantic = np.r_[field.coefficients[:nv],
                     (field.coefficients[edges].sum(axis=1) + 2 * field.coefficients[nv:]) / 4]
    old = np.r_[canonical.harmonic_r, canonical.harmonic_r[edges].mean(axis=1)]
    with path.open('w', encoding='ascii', newline='\n') as stream:
        stream.write('# vtk DataFile Version 3.0\nExperimental semantic scalar; NOT validated fibers\n')
        stream.write('ASCII\nDATASET UNSTRUCTURED_GRID\n')
        stream.write(f'POINTS {len(points)} double\n')
        np.savetxt(stream, points, fmt='%.17g')
        stream.write(f'CELLS {len(cells)} {11 * len(cells)}\n')
        np.savetxt(stream, np.column_stack((np.full(len(cells), 10), cells)), fmt='%d')
        stream.write(f'CELL_TYPES {len(cells)}\n')
        np.savetxt(stream, np.full(len(cells), 24), fmt='%d')
        stream.write(f'POINT_DATA {len(points)}\n')
        for name, values in (('semantic_r', semantic), ('linear_harmonic_r', old)):
            stream.write(f'SCALARS {name} double 1\nLOOKUP_TABLE default\n')
            np.savetxt(stream, values, fmt='%.17g')
        stream.write(f'CELL_DATA {len(cells)}\n')
        for name in ('old_zero_cells', 'near_flat_cells', 'weak_interior_cells',
                     'critical_cell_class', 'sampled_min_gradient', 'semantic_centroid_r',
                     'canonical_parent_tetrahedron_indices'):
            if name not in diagnostics:
                continue
            stream.write(f'SCALARS {name} double 1\nLOOKUP_TABLE default\n')
            np.savetxt(stream, diagnostics[name], fmt='%.17g')


def _write_artifacts(directory, canonical, field, report, diagnostics, inputs, overwrite=False):
    directory = Path(directory)
    if any((directory / name).exists() for name in ARTIFACTS) and not overwrite:
        raise FileExistsError(f'{directory}: use --overwrite to replace known artifacts')
    directory.mkdir(parents=True, exist_ok=True)
    payload = dict(schema_version=1, stage='canonical_semantic_scalar_field',
                   configuration=semantic_configuration(), inputs=inputs,
                   source_geometry_digest=field.source_geometry_digest,
                   coefficient_digest=array_digest(field.coefficients, field.unique_edges),
                   solver=field.solver, counts=dict(coefficients=len(field.coefficients),
                       edges=len(field.unique_edges), zero=len(field.zero_indices), one=len(field.one_indices)),
                   **report)
    finite = bool(np.isfinite(field.coefficients).all())
    with tempfile.TemporaryDirectory(prefix='.semantic-field-', dir=directory) as temporary:
        staging = Path(temporary)
        if finite:
            _write_deterministic_npz(staging / ARTIFACTS[1], coefficients=field.coefficients,
                                     unique_edges=field.unique_edges, **diagnostics)
            _write_vtk(staging / ARTIFACTS[2], canonical, field, diagnostics)
        (staging / ARTIFACTS[0]).write_text(json.dumps(payload, indent=2, sort_keys=True,
                                                     allow_nan=False) + '\n')
        if finite:
            _load_field(canonical, staging)
        # Invalidate any previous completion marker before replacing its arrays.
        if overwrite:
            (directory / ARTIFACTS[0]).unlink(missing_ok=True)
        for name in ARTIFACTS[1:]:
            if finite:
                os.replace(staging / name, directory / name)
            elif overwrite:
                (directory / name).unlink(missing_ok=True)
        os.replace(staging / ARTIFACTS[0], directory / ARTIFACTS[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workflow',nargs='?',choices=['scalar','fibers'],default='scalar')
    parser.add_argument('--anatomical-volume-root', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--refine-from', type=Path,
                        help='Converged, unrefined semantic-field reference directory; one local resolution experiment.')
    parser.add_argument('--scalar-field-root',type=Path)
    parser.add_argument('--extended-anatomical-surface-root',type=Path)
    parser.add_argument('--instance-volume-batch-root',type=Path)
    parser.add_argument('--containment-fit-root',type=Path)
    parser.add_argument('--exclude',action='append',default=[])
    parser.add_argument('--shoes',nargs='*',default=[])
    parser.add_argument('--jobs',type=int,default=1)
    args = parser.parse_args()
    if args.workflow == 'fibers':
        if not 1 <= args.jobs <= 8:
            parser.error('--jobs must be between 1 and 8')
        return _run_fibers(args)
    directory = args.output_root.resolve() / 'reference'
    if any((directory / name).exists() for name in ARTIFACTS) and not args.overwrite:
        parser.error(f'{directory} already contains artifacts; explicit --overwrite required')
    start = time.monotonic()
    canonical = load_canonical_anatomical_volume(args.anatomical_volume_root)
    sources = [args.anatomical_volume_root.resolve() / 'reference' / name
               for name in ('canonical_volume.json', 'canonical_volume.npz')]
    inputs = [dict(path=str(path), sha256=_digest(path)) for path in sources]
    print(f'canonical loading: {time.monotonic()-start:.2f}s', flush=True)
    original = canonical
    refinement = None
    if args.refine_from is not None:
        baseline_directory = args.refine_from.resolve()
        baseline_payload = json.loads((baseline_directory / ARTIFACTS[0]).read_text())
        if baseline_payload.get('refinement'):
            raise ValueError('this bounded experiment accepts only an unrefined baseline')
        baseline_field = _load_field(original, baseline_directory)
        if not baseline_field.solver['converged']:
            raise ValueError('refinement requires a converged baseline')
        with np.load(baseline_directory / ARTIFACTS[1], allow_pickle=False) as archive:
            mask = np.isin(archive['stationary_locations'], ['cell_interior', 'internal_interface'])
            seeds = np.unique(archive['stationary_cell_ids'][mask])
        canonical = refine_scalar_neighborhoods(original, seeds)
        inputs.extend(dict(path=str(baseline_directory / name), sha256=_digest(baseline_directory / name))
                      for name in ARTIFACTS[:2])
        refinement = dict(method='centroid_1_to_4', face_adjacency_rings=1,
            canonical_source_geometry_digest=array_digest(original.volume_vertices, original.tetrahedra),
            seed_parent_indices=seeds.tolist(), refined_parent_indices=canonical.refined_parent_indices.tolist(),
            canonical_tetrahedra=len(original.tetrahedra), auxiliary_tetrahedra=len(canonical.tetrahedra),
            added_vertices=len(canonical.refined_parent_indices), canonical_geometry_modified=False)
        print(f'scalar-only refinement: {refinement}', flush=True)
    field = solve_canonical_semantic_field(canonical)
    audit_start = time.monotonic()
    report, diagnostics = audit_semantic_field(canonical, field)
    if refinement is not None and diagnostics:
        parent_ids = canonical.parent_tetrahedron_indices
        diagnostics['canonical_parent_tetrahedron_indices'] = parent_ids
        stationary_ids = diagnostics['stationary_cell_ids']
        parent = parent_ids[stationary_ids]
        diagnostics['stationary_canonical_parent_ids'] = parent
        points = np.einsum('ni,nij->nj', diagnostics['stationary_barycentric'],
                           canonical.volume_vertices[canonical.tetrahedra[stationary_ids]])
        parent_points = original.volume_vertices[original.tetrahedra[parent]]
        matrices = np.transpose(parent_points[:, 1:] - parent_points[:, :1], (0, 2, 1))
        weights = np.linalg.solve(matrices, (points - parent_points[:, 0])[..., None])[..., 0]
        diagnostics['stationary_canonical_barycentric'] = np.c_[1 - weights.sum(axis=1), weights]
        original_ids = np.arange(len(original.tetrahedra))
        query = np.full((len(original_ids), 4), .25)
        query[canonical.refined_parent_indices] = [1., 0., 0., 0.]
        refined_values = evaluate_semantic_field(field, original_ids, query)[0]
        baseline_values = evaluate_semantic_field(baseline_field, original_ids,
                                                np.full_like(query, .25))[0]
        old_zero = np.all(original.harmonic_r[original.tetrahedra] == 0, axis=1)
        internal = np.isin(diagnostics['stationary_locations'], ['cell_interior', 'internal_interface'])
        refinement.update(original_zero_cells=int(old_zero.sum()),
            original_zero_cells_positive_centroid=int(np.sum(old_zero & (refined_values > 0))),
            maximum_original_centroid_change=float(np.max(abs(refined_values - baseline_values))),
            stationary_canonical_parent_ids=np.unique(parent[internal]).tolist(),
            stationary_points_outside_refined_parents=int(np.sum(internal & ~np.isin(parent, canonical.refined_parent_indices))))
        report['refinement'] = refinement
    print(f'audit: {time.monotonic()-audit_start:.2f}s; {report}', flush=True)
    for source in inputs:
        if _digest(source['path']) != source['sha256']:
            raise RuntimeError('canonical input changed during run')
    writing = time.monotonic()
    _write_artifacts(directory, canonical, field, report, diagnostics, inputs, args.overwrite)
    print(f'artifact writing: {time.monotonic()-writing:.2f}s', flush=True)
    print(f"DONE: {report['status']}; total {time.monotonic()-start:.2f}s; {directory}", flush=True)
    return 0 if report['status'] == 'scalar_candidate' else 1


if __name__ == '__main__':
    raise SystemExit(main())
