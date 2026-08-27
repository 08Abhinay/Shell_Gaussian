# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import os
from tqdm import trange

import numpy as np
import torch
import torch.nn.functional as F

from render import mesh
from render import render
from render import optixutils as ou
from render import regularizer

from .gshell_tets import GShell_Tets

import kaolin

from .mlp import MLP
from foot_prior import (
    FootPriorLoss,
    FootPriorLossConfig,
    PseudoLastPriorConfig,
    PseudoLastPriorLoss,
)


###############################################################################
# Regularizer
###############################################################################

def resolve_tets_path(grid_res):
    module_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(module_dir, "..", ".."))
    candidates = [
        os.path.join(project_root, "data", "tets", f"{grid_res}_tets.npz"),
        os.path.join(project_root, "..", "FootShellGaussian", "data", "tets", f"{grid_res}_tets.npz"),
        os.path.join(project_root, "..", "baselines", "GShell", "data", "tets", f"{grid_res}_tets.npz"),
        os.path.join(os.getcwd(), "data", "tets", f"{grid_res}_tets.npz"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    searched = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        f"Could not find tet grid file for resolution {grid_res}. Searched:\n{searched}"
    )

def compute_sdf_reg_loss(sdf, all_edges):
    sdf_f1x6x2 = sdf[all_edges.reshape(-1)].reshape(-1,2)
    mask = torch.sign(sdf_f1x6x2[...,0]) != torch.sign(sdf_f1x6x2[...,1])
    sdf_f1x6x2 = sdf_f1x6x2[mask]
    sdf_diff = torch.nn.functional.binary_cross_entropy_with_logits(sdf_f1x6x2[...,0], (sdf_f1x6x2[...,1] > 0).float()) + \
            torch.nn.functional.binary_cross_entropy_with_logits(sdf_f1x6x2[...,1], (sdf_f1x6x2[...,0] > 0).float())
    return sdf_diff

###############################################################################
#  Geometry interface
###############################################################################

class GShellTetsGeometry(torch.nn.Module):
    def __init__(self, grid_res, scale, FLAGS, offset=None, tet_init_file=None, extract_from_generative=False):
        super(GShellTetsGeometry, self).__init__()

        self.FLAGS         = FLAGS
        self.grid_res      = grid_res
        self.gshell_tets   = GShell_Tets()
        self.scale         = scale
        self.boxscale      = torch.tensor(FLAGS.boxscale).view(1, 3).cuda()

        with torch.no_grad():
            self.optix_ctx = ou.OptiXContext()

            if tet_init_file is None:
                tet_init_file = resolve_tets_path(self.grid_res)
                tets = np.load(tet_init_file)
            else:
                tets = np.load(tet_init_file)
            print(f'using resolution {self.grid_res} from {tet_init_file}')
            self.verts    = torch.tensor(tets['vertices'], dtype=torch.float32, device='cuda')
            self.original_verts = self.verts.clone() if extract_from_generative else None
            self.verts    = self.verts - self.verts.mean(dim=0)
            self.verts    = self.verts * scale * self.boxscale
            self.indices  = torch.tensor(tets['indices'], dtype=torch.long, device='cuda')
            self.generate_edges()

            if extract_from_generative:
                self.sorted_tetedges = torch.tensor(tets['tet_edges'], dtype=torch.long, device='cuda')
                vertices = torch.tensor(tets['vertices'], dtype=torch.float32, device='cuda')
                vertices_unique = vertices.view(-1).unique()
                dx = (vertices_unique[1] - vertices_unique[0]) / 2.0 ### denser grid for edge + tet features
                vertices_discretized = (
                    ((vertices - vertices.min()) / dx)
                ).long()
                self.verts_discretized = vertices_discretized.long().float() ### used to identify where to store edge + tet features

            if offset is None:
                offset = 0.0
            else:
                offset = torch.tensor(offset).cuda().view(1, 3)
            self.offset = offset

        self.foot_prior_loss = None
        self.last_foot_prior_stats = {}
        canonical_grid_points_world = self.verts
        if torch.is_tensor(self.offset):
            canonical_grid_points_world = canonical_grid_points_world + self.offset.to(
                device=canonical_grid_points_world.device,
                dtype=canonical_grid_points_world.dtype,
            )
        elif self.offset != 0.0:
            canonical_grid_points_world = canonical_grid_points_world + float(self.offset)
        canonical_y_coords = canonical_grid_points_world[:, 1]
        canonical_y_bottom = canonical_y_coords.max()
        canonical_y_top = canonical_y_coords.min()
        self.bottom_msdf_slab_info = {
            "canonical_y_bottom": float(canonical_y_bottom.item()),
            "canonical_y_top": float(canonical_y_top.item()),
            "canonical_height": float((canonical_y_bottom - canonical_y_top).item()),
        }
        if getattr(self.FLAGS, "use_foot_prior", False):
            self.foot_prior_loss = FootPriorLoss(
                FootPriorLossConfig.from_flags(self.FLAGS),
                device=torch.device("cuda"),
            )
            print("Loaded neutral foot prior:")
            print(f"  SDF: {self.FLAGS.foot_prior_sdf_path}")
            print(f"  alignment: {self.FLAGS.foot_prior_alignment_path}")

        self.pseudo_last_prior_loss = None
        self.last_pseudo_last_prior_stats = {}
        self.last_bottom_msdf_conditioning_stats = {}
        if getattr(self.FLAGS, "use_pseudo_last_prior", False):
            if getattr(self.FLAGS, "pseudo_last_prior_mode", "cross_section") == "cross_section" and not self.FLAGS.use_sdf_mlp:
                raise ValueError("The cross-section pseudo-last prior requires use_sdf_mlp=True")
            self.pseudo_last_prior_loss = PseudoLastPriorLoss(
                PseudoLastPriorConfig.from_flags(self.FLAGS),
                device=torch.device("cuda"),
            )
            grid_points_world = self.verts
            if torch.is_tensor(self.offset):
                grid_points_world = grid_points_world + self.offset.to(
                    device=grid_points_world.device,
                    dtype=grid_points_world.dtype,
                )
            elif self.offset != 0.0:
                grid_points_world = grid_points_world + float(self.offset)
            grid_stats = self.pseudo_last_prior_loss.precompute_grid_keep_weights(
                grid_points_world,
                tet_edges=self.all_edges,
            )
            print("Loaded pseudo-last topology prior:")
            print(f"  SDF: {self.FLAGS.pseudo_last_sdf_path}")
            print(f"  sections: {self.FLAGS.pseudo_last_sections_path}")
            print(
                "  cross-section pool: "
                f"material={int(grid_stats.get('xsec_material_pool_count', 0))}, "
                f"sole={int(grid_stats.get('xsec_sole_pool_count', 0))}, "
                f"plantar={int(grid_stats.get('xsec_plantar_pool_count', 0))}, "
                f"sidewall={int(grid_stats.get('xsec_sidewall_pool_count', 0))}, "
                f"empty={int(grid_stats.get('xsec_empty_pool_count', 0))}, "
                f"surface={int(grid_stats.get('xsec_surface_pool_count', 0))}"
            )
            print(
                "  cross-section grid mSDF candidates: "
                f"{int(grid_stats.get('xsec_grid_msdf_candidate_count', 0))} "
                f"(sole={int(grid_stats.get('xsec_grid_sole_candidate_count', 0))}, "
                f"plantar={int(grid_stats.get('xsec_grid_plantar_candidate_count', 0))}, "
                f"sidewall={int(grid_stats.get('xsec_grid_sidewall_candidate_count', 0))})"
            )
            print(
                "  cross-section grid candidate h: "
                f"p50={float(grid_stats.get('xsec_grid_h_p50', 0.0)):.5f}, "
                f"p99={float(grid_stats.get('xsec_grid_h_p99', 0.0)):.5f}, "
                f"far_below={int(grid_stats.get('xsec_grid_far_below_count', 0))}"
            )

        if self.FLAGS.use_sdf_mlp:
            self.sdf    = torch.nn.Parameter(torch.zeros_like(self.verts[:, 0]), requires_grad=True) ## placeholder
            self.register_parameter('sdf', self.sdf)
            self.sdf_net = MLP(
                skip_in=self.FLAGS.skip_in,
                n_freq=self.FLAGS.n_freq,
                n_hidden=self.FLAGS.n_hidden,
                d_hidden=self.FLAGS.d_hidden,
                use_float16=self.FLAGS.use_float16
            )
            self.sdf_net.cuda()

            optimizer = torch.optim.Adam(self.sdf_net.parameters(), lr=1e-3)
            for _ in trange(self.FLAGS.sdf_mlp_pretrain_steps):
                scaled_verts = self.verts / self.boxscale
                loss = (self.sdf_net(self.verts) - (scaled_verts.norm(dim=1, keepdim=True) - self.FLAGS.sphere_init_norm)).pow(2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print('sdf net trained with loss:', loss)

        else:
            # Random init
            if not self.FLAGS.sphere_init:
                sdf = torch.rand_like(self.verts[:,0]) - 0.1
            else:
                scaled_verts = self.verts / self.boxscale
                sdf = scaled_verts.norm(dim=1) - 0.5
            self.sdf    = torch.nn.Parameter(sdf.clone().detach(), requires_grad=True)
            self.register_parameter('sdf', self.sdf)


        if self.FLAGS.use_msdf_mlp:
            self.msdf    = torch.nn.Parameter(torch.zeros_like(self.verts[:, 0]), requires_grad=True) ## placeholder
            self.register_parameter('msdf', self.msdf)
            self.msdf_net = MLP(
                skip_in=self.FLAGS.skip_in,
                n_freq=self.FLAGS.n_freq,
                n_hidden=self.FLAGS.n_hidden,
                d_hidden=self.FLAGS.d_hidden,
                use_float16=self.FLAGS.use_float16
            )
            self.msdf_net.cuda()
            optimizer = torch.optim.Adam(self.msdf_net.parameters(), lr=1e-3)
            for _ in trange(100):
                scaled_verts = self.verts / self.boxscale
                loss = (self.msdf_net(self.verts) - 0.1).pow(2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print('sdf net trained with loss:', loss)
            del optimizer
        else:
            msdf         = (torch.rand_like(self.verts[:,0]) - 0.01).clamp(-1, 1)
            self.msdf    = torch.nn.Parameter(msdf.clone().detach(), requires_grad=True)
            self.register_parameter('msdf', self.msdf)

        self.deform = torch.nn.Parameter(torch.zeros_like(self.verts), requires_grad=True)
        self.register_parameter('deform', self.deform)

        self.clamp_deform()

    @torch.no_grad()
    def generate_edges(self):
        with torch.no_grad():
            edges = torch.tensor([0,1,0,2,0,3,1,2,1,3,2,3], dtype = torch.long, device = "cuda")
            all_edges = self.indices[:,edges].reshape(-1,2)
            all_edges_sorted = torch.sort(all_edges, dim=1)[0]
            self.all_edges = torch.unique(all_edges_sorted, dim=0)
            self.max_displacement = 1.0 / self.grid_res * self.scale / 2.1

    @torch.no_grad()
    def getAABB(self):
        return torch.min(self.verts, dim=0).values, torch.max(self.verts, dim=0).values

    @torch.no_grad()
    def clamp_deform(self):
        if not self.FLAGS.use_tanh_deform:
            self.deform.data[:] = self.deform.clamp(-1.0, 1.0)
        self.msdf.data[:] = self.msdf.clamp(-2.0, 2.0)

    def getMesh_from_augmented_grid_withocc(self, material, sdf_sign, sdf_coeff, msdf_sign, occgrid):
        # Run DM tet to get a base mesh
        v_deformed = self.verts + self.max_displacement * self.deform
        if self.FLAGS.use_sdf_mlp:
            sdf = self.sdf_net(v_deformed)
        else:
            sdf = self.sdf

        verts, faces, uvs, uv_idx, v_tng, v_pos_original, tet_gidx, v_msdf, msdf_vert_original = self.gshell_tets.marching_from_auggrid(
            v_deformed, sdf_sign, self.indices,
            self.sorted_tetedges, sdf_coeff, self.verts_discretized, 
            msdf_sign, 
            occgrid)
        imesh = mesh.Mesh(verts, faces, v_tex=uvs, t_tex_idx=uv_idx, material=material)

        # Run mesh operations to generate tangent space
        imesh = mesh.auto_normals(imesh)
        imesh = mesh.compute_tangents(imesh, v_tng=v_tng)
        return {
            'imesh': imesh,
            'sdf': sdf,
            'v_msdf': v_msdf,
        }

    def _resolve_bottom_msdf_anchor_y(
        self,
        sdf: torch.Tensor,
        grid_points_world: torch.Tensor,
    ):
        canonical_y_bottom = float(self.bottom_msdf_slab_info["canonical_y_bottom"])
        canonical_y_top = float(self.bottom_msdf_slab_info["canonical_y_top"])
        anchor_mode = str(getattr(self.FLAGS, "bottom_msdf_anchor_mode", "canonical_grid_bottom"))
        anchor_stats = {
            "anchor_mode": anchor_mode,
            "anchor_y": canonical_y_bottom,
            "solid_point_count": 0,
            "anchor_quantile": float(getattr(self.FLAGS, "bottom_msdf_anchor_quantile", 0.995)),
            "anchor_fallback_used": 0.0,
        }

        if anchor_mode == "canonical_grid_bottom":
            return canonical_y_bottom, anchor_stats

        if anchor_mode == "canonical_fixed_y":
            anchor_y = float(getattr(self.FLAGS, "bottom_msdf_anchor_y", canonical_y_bottom))
            anchor_y = min(max(anchor_y, canonical_y_top), canonical_y_bottom)
            anchor_stats["anchor_y"] = anchor_y
            return anchor_y, anchor_stats

        if anchor_mode == "sdf_solid_quantile":
            solid_mask = sdf.reshape(-1) > 0
            solid_point_count = int(solid_mask.sum().item())
            anchor_stats["solid_point_count"] = solid_point_count
            if solid_point_count == 0:
                anchor_stats["anchor_fallback_used"] = 1.0
                return canonical_y_bottom, anchor_stats

            solid_y = grid_points_world.reshape(-1, 3)[:, 1][solid_mask]
            quantile = float(anchor_stats["anchor_quantile"])
            anchor_y = float(torch.quantile(solid_y, quantile).item())
            anchor_y = min(max(anchor_y, canonical_y_top), canonical_y_bottom)
            anchor_stats["anchor_y"] = anchor_y
            return anchor_y, anchor_stats

        raise ValueError(f"Unknown bottom_msdf_anchor_mode: {anchor_mode}")

    def _condition_bottom_band_msdf_for_marching(
        self,
        raw_msdf: torch.Tensor,
        sdf: torch.Tensor,
        grid_points_world: torch.Tensor,
    ):
        canonical_y_bottom = float(self.bottom_msdf_slab_info["canonical_y_bottom"])
        canonical_height = float(self.bottom_msdf_slab_info["canonical_height"])
        stats = {
            "enabled": 0.0,
            "canonical_y_bottom": canonical_y_bottom,
            "canonical_height": canonical_height,
            "anchor_mode": str(getattr(self.FLAGS, "bottom_msdf_anchor_mode", "canonical_grid_bottom")),
            "anchor_y": canonical_y_bottom,
            "solid_point_count": 0,
            "anchor_quantile": float(getattr(self.FLAGS, "bottom_msdf_anchor_quantile", 0.995)),
            "anchor_fallback_used": 0.0,
            "full_count": 0,
            "blend_count": 0,
            "changed_count": 0,
            "active_point_count": 0,
        }

        if not getattr(self.FLAGS, "use_bottom_msdf_conditioning", False):
            return raw_msdf, stats

        margin = float(getattr(self.FLAGS, "bottom_msdf_margin", 0.01))
        protect_height = float(getattr(self.FLAGS, "bottom_msdf_protect_height", 0.012))
        blend_height = float(getattr(self.FLAGS, "bottom_msdf_blend_height", 0.008))
        anchor_y, anchor_stats = self._resolve_bottom_msdf_anchor_y(
            sdf=sdf,
            grid_points_world=grid_points_world,
        )
        stats.update(anchor_stats)

        raw_msdf_flat = raw_msdf.reshape(-1)
        grid_y_coords = grid_points_world.reshape(-1, 3)[:, 1]
        distance_up_from_bottom = raw_msdf_flat.new_tensor(anchor_y) - grid_y_coords

        full_mask = distance_up_from_bottom <= protect_height
        blend_mask = (
            (distance_up_from_bottom > protect_height)
            & (distance_up_from_bottom <= protect_height + blend_height)
        )

        weights = torch.zeros_like(raw_msdf_flat)
        weights[full_mask] = 1.0
        if blend_height > 0.0 and torch.any(blend_mask):
            fade_progress = (distance_up_from_bottom[blend_mask] - protect_height) / blend_height
            weights[blend_mask] = 1.0 - fade_progress.clamp(0.0, 1.0)

        safe_positive_msdf = raw_msdf_flat.clamp(min=margin)
        conditioned_msdf_flat = (
            weights * safe_positive_msdf
            + (1.0 - weights) * raw_msdf_flat
        )
        conditioned_msdf = conditioned_msdf_flat.reshape_as(raw_msdf)

        full_count = int(full_mask.sum().item())
        blend_count = int(blend_mask.sum().item())
        changed_count = int((conditioned_msdf_flat != raw_msdf_flat).sum().item())

        stats["enabled"] = 1.0
        stats["full_count"] = full_count
        stats["blend_count"] = blend_count
        stats["changed_count"] = changed_count
        stats["active_point_count"] = full_count + blend_count

        return conditioned_msdf, stats

    def getMesh(self, material, iteration=None):
        v_deformed = self.verts + self.max_displacement * self.deform
        if self.FLAGS.use_sdf_mlp:
            sdf = self.sdf_net(v_deformed)
        else:
            sdf = self.sdf
        

        if self.FLAGS.use_msdf_mlp:
            msdf = self.msdf_net(v_deformed)
        else:
            msdf = self.msdf

        v_deformed = v_deformed + self.offset
        (
            msdf_for_marching,
            self.last_bottom_msdf_conditioning_stats,
        ) = self._condition_bottom_band_msdf_for_marching(
            raw_msdf=msdf,
            sdf=sdf,
            grid_points_world=v_deformed,
        )

        verts, faces, uvs, uv_idx, v_tng, extra = self.gshell_tets(
            v_deformed, sdf, msdf_for_marching, self.indices)
        imesh = mesh.Mesh(verts, faces, v_tex=uvs, t_tex_idx=uv_idx, material=material)

        with torch.no_grad():
            ou.optix_build_bvh(self.optix_ctx, imesh.v_pos.contiguous(), imesh.t_pos_idx.int(), rebuild=1)

        # Run mesh operations to generate tangent space
        imesh = mesh.auto_normals(imesh)
        return_dict = {
            'imesh': imesh,
            'sdf': sdf,
            'msdf': extra['msdf'],
            'msdf_watertight': extra['msdf_watertight'],
            'msdf_boundary': extra['msdf_boundary'],
            'n_verts_watertight': extra['n_verts_watertight'],
            'vertices_watertight': extra['vertices_watertight'],
            'faces_watertight': extra['faces_watertight'],
            'grid_msdf': msdf,
            'grid_msdf_raw': msdf,
            'grid_msdf_for_marching': msdf_for_marching,
            'grid_sdf': sdf,
            'grid_points_world': v_deformed,
        }

        if self.FLAGS.visualize_watertight:
            imesh_watertight = mesh.Mesh(extra['vertices_watertight'], extra['faces_watertight'], v_tex=None, t_tex_idx=None, material=material)
            imesh_watertight = mesh.auto_normals(imesh_watertight)
            return_dict['imesh_watertight'] = imesh_watertight
        return return_dict

    def render(self, glctx, target, lgt, opt_material, bsdf=None, denoiser=None, shadow_scale=1.0,
            iteration=None,
            use_uv=False):
        opt_mesh_dict = self.getMesh(opt_material, iteration=iteration)
        opt_mesh = opt_mesh_dict['imesh']
        opt_mesh_watertight = opt_mesh_dict['imesh_watertight'] if 'imesh_watertight' in opt_mesh_dict else None
        if opt_mesh.v_pos.size(0) != 0 and opt_mesh.t_pos_idx.size(0) != 0:
            sampled_pts = kaolin.ops.mesh.sample_points(opt_mesh.v_pos[None,...], opt_mesh.t_pos_idx, 50000)[0][0]
            opt_mesh_dict['sampled_pts'] = sampled_pts
        else:
            opt_mesh_dict['sampled_pts'] = None
    
        extra_dict = {
            'msdf': opt_mesh_dict['msdf'],
        }
        opt_mesh_dict['buffers'] = render.render_mesh(
            self.FLAGS, glctx, opt_mesh, target['mvp'], target['campos'], lgt, target['resolution'], spp=target['spp'], 
            msaa=True, background=target['background'], bsdf=bsdf, use_uv=use_uv,
            optix_ctx=self.optix_ctx, denoiser=denoiser, shadow_scale=shadow_scale,
            extra_dict=extra_dict)
        if self.FLAGS.visualize_watertight:
            opt_mesh_dict['buffers_watertight'] = render.render_mesh(
                self.FLAGS, glctx, opt_mesh_watertight, target['mvp'], target['campos'], lgt, target['resolution'], spp=target['spp'], 
                msaa=True, background=target['background'], bsdf=bsdf, use_uv=use_uv,
                optix_ctx=self.optix_ctx, denoiser=denoiser, shadow_scale=shadow_scale,
                extra_dict=extra_dict)
        return opt_mesh_dict

    def tick(self, glctx, target, lgt, opt_material, loss_fn, iteration, denoiser):

        t_iter = iteration / self.FLAGS.iter

        # ==============================================================================================
        #  Render optimizable object with identical conditions
        # ==============================================================================================
        shadow_ramp = min(iteration / 1000, 1.0) ### set occlusion ray influence
        if denoiser is not None: denoiser.set_influence(shadow_ramp)
        opt_mesh_dict = self.render(glctx, target, lgt, opt_material, 
            denoiser=denoiser,
            shadow_scale=shadow_ramp,
            iteration=iteration)
        buffers = opt_mesh_dict['buffers']

        # ==============================================================================================
        #  Compute loss
        # ==============================================================================================

        with torch.no_grad():
            # Image-space loss, split into a coverage component and a color component
            color_ref = target['img']
            gt_mask = color_ref[..., 3:]

        img_loss = F.mse_loss(buffers['shaded'][..., 3:], color_ref[..., 3:]) 
        img_loss = img_loss + loss_fn(buffers['shaded'][..., 0:3] * color_ref[..., 3:], color_ref[..., 0:3] * color_ref[..., 3:])


        img_loss = img_loss + 5e-1 * F.l1_loss(buffers['msdf_image'].clamp(min=0) * (gt_mask == 0).float(), torch.zeros_like(gt_mask))
        img_loss = img_loss + 5e-1 * F.l1_loss(buffers['msdf_image'].clamp(max=0) * (gt_mask == 1).float(), torch.ones_like(gt_mask))

        if self.FLAGS.use_img_2nd_layer:
            color_ref_2nd = target['img_second']
            img_loss = img_loss + F.mse_loss(buffers['shaded_second'][..., 3:], color_ref_2nd[..., 3:]) 
            img_loss = img_loss + loss_fn(buffers['shaded_second'][..., 0:3] * color_ref_2nd[..., 3:], color_ref_2nd[..., 0:3] * color_ref_2nd[..., 3:])

        if self.FLAGS.use_depth:
            depth_loss_scale = 100.
            depth_loss = depth_loss_scale * ((buffers['invdepth'][:, :, :, :1] - target['invdepth'][:, :, :, :1]).abs()).mean()

            if self.FLAGS.use_depth_2nd_layer:
                depth_loss += 0.1 * depth_loss_scale * ((buffers['invdepth_second'][:, :, :, :1] - target['invdepth_second'][:, :, :, :1]).abs()).mean()
        else:
            depth_loss = torch.tensor(0., device=img_loss.device)

        # Eikonal
        if self.FLAGS.use_sdf_mlp and self.FLAGS.use_eikonal and opt_mesh_dict['sampled_pts'] is not None:
            v = opt_mesh_dict['sampled_pts'].detach()
            v.requires_grad = True

            sdf_eik = self.sdf_net(v)
            if self.FLAGS.eikonal_scale is None:
                ### Default hardcoded Eikonal loss schedule
                if iteration < 500:
                    eik_coeff = 3e-1
                elif iteration < 1000:
                    eik_coeff = 1e-1
                elif iteration < 2000:
                    eik_coeff = 1e-1
                else:
                    eik_coeff = 1e-2
            else:
                eik_coeff = self.FLAGS.eikonal_scale

            eik_loss = eik_coeff * (
                torch.autograd.grad(sdf_eik.sum(), v, create_graph=True)[0].pow(2).sum(dim=-1).sqrt() - 1
            ).pow(2).mean()
        else:
            eik_loss = torch.tensor(0., device=img_loss.device)

        if self.FLAGS.use_mesh_msdf_reg:
            mesh_msdf_regscale = (64 / self.grid_res) ** 3 # scale inversely proportional to grid_res^3
            eps = 1e-3
            open_scale = self.FLAGS.msdf_reg_open_scale
            close_scale = self.FLAGS.msdf_reg_close_scale

            if open_scale > 0:
                mesh_msdf = opt_mesh_dict['msdf'].reshape(-1)
                mesh_msdf_reg_loss = open_scale * mesh_msdf_regscale * F.huber_loss(
                    mesh_msdf.clamp(min=-eps),
                    torch.full_like(mesh_msdf, -eps),
                    reduction='sum'
                )
            else:
                mesh_msdf_reg_loss = torch.tensor(0., device=img_loss.device)

            if close_scale != 0:
                with torch.no_grad():
                    visible_verts = (opt_mesh_dict['imesh'].t_pos_idx[buffers['visible_triangles']]).unique()
                    visible_boundary_verts = visible_verts[visible_verts >= opt_mesh_dict['n_verts_watertight']] - opt_mesh_dict['n_verts_watertight']
                    visible_boundary_mask = torch.zeros(opt_mesh_dict['msdf_boundary'].size(0)).cuda()
                    visible_boundary_mask[visible_boundary_verts] = 1
                    visible_boundary_mask = visible_boundary_mask.bool()

                boundary_msdf = opt_mesh_dict['msdf_boundary']
                boundary_msdf = boundary_msdf[visible_boundary_mask].reshape(-1)
                if boundary_msdf.numel() > 0:
                    mesh_msdf_reg_loss += close_scale * mesh_msdf_regscale * F.huber_loss(
                        boundary_msdf.clamp(max=eps),
                        torch.full_like(boundary_msdf, eps),
                        reduction='sum'
                    )
        else:
            mesh_msdf_reg_loss = torch.tensor(0., device=img_loss.device)

        # SDF regularizer
        sdf_weight = self.FLAGS.sdf_regularizer - (self.FLAGS.sdf_regularizer - 0.01) * min(1.0, 4.0 * t_iter)
        sdf_reg_loss = compute_sdf_reg_loss(opt_mesh_dict['sdf'], self.all_edges).mean() * sdf_weight

        # Monochrome shading regularizer
        if 'diffuse_light' not in buffers:
            monochrome_loss = torch.zeros_like(img_loss)
        else:
            monochrome_loss = regularizer.shading_loss(buffers['diffuse_light'], buffers['specular_light'], color_ref, self.FLAGS.lambda_diffuse, self.FLAGS.lambda_specular)

        # Material smoothness regularizer
        mtl_smooth_loss = regularizer.material_smoothness_grad(
            buffers['kd_grad'], buffers['ks_grad'], buffers['normal_grad'], 
            lambda_kd=self.FLAGS.lambda_kd, lambda_ks=self.FLAGS.lambda_ks, lambda_nrm=self.FLAGS.lambda_nrm)

        # Chroma regularizer
        chroma_loss = regularizer.chroma_loss(buffers['kd'], color_ref, self.FLAGS.lambda_chroma)
        assert 'perturbed_nrm' not in buffers # disable normal map in first pass


        foot_prior_reg_loss = torch.tensor(0., device=img_loss.device)
        self.last_foot_prior_stats = {}
        if self.foot_prior_loss is not None:
            surface_points = opt_mesh_dict['sampled_pts']
            if surface_points is None and opt_mesh_dict['imesh'].v_pos.numel() > 0:
                surface_points = opt_mesh_dict['imesh'].v_pos
            foot_prior_reg_loss, self.last_foot_prior_stats = self.foot_prior_loss(
                iteration=iteration,
                surface_points=surface_points,
                watertight_points=opt_mesh_dict.get('vertices_watertight'),
                watertight_msdf=opt_mesh_dict.get('msdf_watertight'),
            )

        pseudo_last_prior_reg_loss = torch.tensor(0., device=img_loss.device)
        self.last_pseudo_last_prior_stats = {}
        if self.pseudo_last_prior_loss is not None:
            def query_shell_sdf(points_world):
                query_points = points_world
                if torch.is_tensor(self.offset):
                    query_points = query_points - self.offset.to(
                        device=query_points.device,
                        dtype=query_points.dtype,
                    )
                return self.sdf_net(query_points).reshape(-1)

            pseudo_last_prior_reg_loss, self.last_pseudo_last_prior_stats = self.pseudo_last_prior_loss(
                iteration=iteration,
                shell_sdf_query_fn=query_shell_sdf,
                grid_msdf=opt_mesh_dict.get('grid_msdf'),
            )

        geo_reg_loss = (
            sdf_reg_loss
            + eik_loss
            + mesh_msdf_reg_loss
            + foot_prior_reg_loss
            + pseudo_last_prior_reg_loss
        )
        shading_reg_loss =  monochrome_loss + mtl_smooth_loss + chroma_loss
        reg_loss = geo_reg_loss + shading_reg_loss

        return img_loss, depth_loss, reg_loss
