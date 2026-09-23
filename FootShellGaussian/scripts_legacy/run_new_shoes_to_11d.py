#!/usr/bin/env python3
"""Prepare the additional reviewed right shoes, then run the existing pipeline to 11-D.

The two phases are deliberately separate: a tmux shell can run ``prepare`` and
then ``continue`` with ``&&``. Existing golden-set shoes and batches are never
overwritten; all geometric outputs for these shoes live under one new run root.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
PYTHON = Path('/home/ab5298/anaconda3/envs/shellgaussianenv/bin/python')
BLENDER = Path('/home/ab5298/anaconda3/envs/shellgaussianenv/bin/blender')
SOURCE_ROOT = Path('/home/ab5298/dataset/datasets/external/golden_set_eval_glb/curated_subsets/footbed_clean')
PROCESSED_ROOT = Path('/home/ab5298/dataset/datasets/processed/gshell/golden_set_evaluation')
STABLE_ROOT = Path('/home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation')
FOOT_MODEL = WORKSPACE_ROOT / 'baselines/SUPR/data/supr_male_right_foot.npy'
BODY_MODEL = WORKSPACE_ROOT / 'baselines/SUPR/data/supr_male.npy'
DATASET_PIPELINE = WORKSPACE_ROOT / 'dataset_tools_blender/pipeline.py'
OLD_MANIFEST = WORKSPACE_ROOT / 'dataset_tools_blender/golden_set_evaluation_manifest.json'

# Axes are specified in Blender coordinates after glTF import, not in the
# source viewer's Y-up coordinates. The importer maps source +Y to Blender +Z
# and source +Z to Blender -Y. All shoes have +X toes except adidas (-X).
NEW_MODELS = {
    'adidas_substance.glb': ('-X', '-Y', 'Z'),
    'rtfktchallenge_-_golden_by_franz_vega.glb': ('X', 'Y', 'Z'),
    **{f'sneaker_{number}.glb': ('X', 'Y', 'Z') for number in range(1, 8)},
    'sneaker_8.glb': ('-Y', 'X', 'Z'),
    'sneaker_b33.glb': ('X', 'Y', 'Z'),
}
SHOES = tuple(sorted(re.sub(r'[^a-z0-9]+', '_', Path(model).stem.lower()).strip('_')
                     for model in NEW_MODELS))
THREAD_ENV = {key: '1' for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
    'MKL_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
    'NUMEXPR_NUM_THREADS')}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def _command(args: list[str], *, cwd: Path = PROJECT_ROOT,
             visible_gpus: list[str] | None = None) -> None:
    print('[pipeline]', ' '.join(args), flush=True)
    environment = {**os.environ, **THREAD_ENV}
    if visible_gpus is not None:
        environment['CUDA_VISIBLE_DEVICES'] = ','.join(visible_gpus)
    subprocess.run(args, cwd=cwd, env=environment, check=True)


def _shoe_manifest() -> dict:
    original = json.loads(OLD_MANIFEST.read_text(encoding='utf-8'))
    records = []
    for model, (length, width, up) in sorted(NEW_MODELS.items()):
        source = SOURCE_ROOT / model
        if not source.is_file():
            raise FileNotFoundError(source)
        name = re.sub(r'[^a-z0-9]+', '_', source.stem.lower()).strip('_')
        records.append(dict(name=name, model=model, sha256=_sha256(source),
            reviewed=True, shoe_profile='normal',
            source_axes=dict(length=length, width=width, up=up),
            selection=dict(mode='all'), mirror_width=False))
    return dict(version=1, description='Eleven additional right-shoe GLBs; reviewed Blender-import axes',
        inventory_policy='listed_subset',
        horizontal_alignment=original['horizontal_alignment'], shoes=records)


def prepare(args: argparse.Namespace) -> None:
    args.run_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.run_root / 'new_shoe_dataset_manifest.json'
    manifest = _shoe_manifest()
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('existing run manifest differs from current source GLBs')
    else:
        _write_json(manifest_path, manifest)
    gpu_ids = args.gpus.split(',')
    if not gpu_ids or any(not gpu.isdigit() for gpu in gpu_ids):
        raise ValueError('--gpus must be a comma-separated list of GPU IDs')
    missing = [name for name in SHOES if not (PROCESSED_ROOT / name).exists()]
    for index, name in enumerate(missing):
        _command([str(PYTHON), str(DATASET_PIPELINE), 'build', '--shoe', name,
            '--gpu', gpu_ids[index % len(gpu_ids)], '--source-root', str(SOURCE_ROOT),
            '--manifest', str(manifest_path), '--output-root', str(PROCESSED_ROOT),
            '--blender', str(BLENDER)], cwd=WORKSPACE_ROOT)
    records = {record['name']: record for record in manifest['shoes']}
    for name in SHOES:
        directory = PROCESSED_ROOT / name
        for filename in ('reference_mesh.ply', 'blender_canonicalization.json', 'transforms.json'):
            if not (directory / filename).is_file():
                raise FileNotFoundError(directory / filename)
        metadata = json.loads((directory / 'blender_canonicalization.json').read_text())
        expected = records[name]
        if (metadata.get('shoe') != name or metadata.get('source_sha256') != expected['sha256']
                or metadata.get('canonical_geometry', {}).get('source_axes') != expected['source_axes']):
            raise ValueError(f'{name}: processed geometry does not match the current source and axes')
        if len(list((directory / 'image').glob('img[0-9][0-9][0-9].jpg'))) != 180:
            raise ValueError(f'{name}: expected 180 rendered views')
    _write_json(args.run_root / 'dataset_preparation.json', dict(
        status='prepared', shoes=list(SHOES), manifest_sha256=_sha256(manifest_path),
        processed_root=str(PROCESSED_ROOT), gpu_list=args.gpus.split(',')))
    print(f'[prepared] {len(SHOES)} shoes; {args.run_root / "dataset_preparation.json"}', flush=True)


def continue_pipeline(args: argparse.Namespace) -> None:
    gpu_ids = args.gpus.split(',')
    if not gpu_ids or any(not gpu.isdigit() for gpu in gpu_ids) or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError('--gpus must list distinct physical GPU IDs')
    if args.jobs > 8 * len(gpu_ids):
        raise ValueError('at most eight shoe workers may be assigned to each GPU')
    marker = args.run_root / 'dataset_preparation.json'
    manifest = args.run_root / 'new_shoe_dataset_manifest.json'
    record = json.loads(marker.read_text())
    if (record.get('status') != 'prepared' or record.get('shoes') != list(SHOES)
            or record.get('manifest_sha256') != _sha256(manifest)
            or json.loads(manifest.read_text()) != _shoe_manifest()):
        raise ValueError('dataset preparation marker or source manifest is not valid')
    for name in SHOES:
        for filename in ('reference_mesh.ply', 'blender_canonicalization.json'):
            if not (PROCESSED_ROOT / name / filename).is_file():
                raise FileNotFoundError(PROCESSED_ROOT / name / filename)
    for required in (FOOT_MODEL, BODY_MODEL, STABLE_ROOT / 'anatomical_volume/reference/canonical_volume.json',
                     STABLE_ROOT / 'anatomical_fibers/converged/reference/semantic_field.json'):
        if not required.is_file():
            raise FileNotFoundError(required)

    root = args.run_root
    completed_before_containment = [
        'run_shoe_preparation.py', 'run_alignment.py', 'run_cavity_analysis.py'
    ]
    if args.resume_containment:
        saved = json.loads((root / 'pipeline_status.json').read_text())
        if (saved.get('status') != 'stopped_by_user'
                or saved.get('completed_stages') != completed_before_containment
                or saved.get('shoes') != list(SHOES)):
            raise ValueError('run is not a verified stopped containment-stage run')
        prior_artifacts = (
            ('shoe_preparation', 'shoe_preparation.json'),
            ('support_fit', 'support_fit.json'),
            ('cavity_analysis', 'cavity_analysis.json'),
        )
        for name in SHOES:
            for stage_dir, artifact in prior_artifacts:
                if not (root / stage_dir / name / artifact).is_file():
                    raise FileNotFoundError(root / stage_dir / name / artifact)
            if (root / 'containment_fit' / name).exists():
                raise FileExistsError(f'{name}: containment output already exists')
    stages = completed_before_containment.copy() if args.resume_containment else []

    def record(stage: str) -> None:
        stages.append(stage)
        _write_json(root / 'pipeline_status.json', dict(status='running', completed_stages=stages,
            shoes=list(SHOES), current_stage=stage))

    def execute(stage: str, argv: list[str]) -> None:
        _command([str(PYTHON), str(PROJECT_ROOT / 'scripts' / stage), *argv],
            visible_gpus=gpu_ids)
        record(stage)

    def execute_parallel(stage: str, arguments) -> None:
        log_dir = root / 'logs' / Path(stage).stem
        log_dir.mkdir(parents=True, exist_ok=True)
        command_base = [str(PYTHON), str(PROJECT_ROOT / 'scripts' / stage)]
        environment = {**os.environ, **THREAD_ENV}
        shoe_gpu = {name: gpu_ids[index % len(gpu_ids)]
                    for index, name in enumerate(SHOES)}

        def run_one(name: str) -> tuple[str, int]:
            command = [*command_base, *arguments(name)]
            with (log_dir / f'{name}.log').open('w', encoding='utf-8') as log:
                result = subprocess.run(command, cwd=PROJECT_ROOT,
                    env={**environment, 'CUDA_VISIBLE_DEVICES': shoe_gpu[name]},
                    stdout=log, stderr=subprocess.STDOUT, check=False)
            return name, result.returncode

        print(f'[stage] {stage}: {len(SHOES)} shoes, up to {args.jobs} workers '
              f'on GPUs {gpu_ids}', flush=True)
        failures = []
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(run_one, name): name for name in SHOES}
            for future in as_completed(futures):
                name, returncode = future.result()
                print(f'[{"ok" if returncode == 0 else "failed"}] {stage}: {name}', flush=True)
                if returncode != 0:
                    failures.append(name)
        if failures:
            raise RuntimeError(f'{stage} failed for {sorted(failures)}; inspect {log_dir}')
        record(stage)

    if not args.resume_containment:
        execute_parallel('run_shoe_preparation.py', lambda name: [
            '--shoe-mesh', str(PROCESSED_ROOT/name/'reference_mesh.ply'),
            '--canonicalization', str(PROCESSED_ROOT/name/'blender_canonicalization.json'),
            '--output-dir', str(root/'shoe_preparation'/name)])
        execute_parallel('run_alignment.py', lambda name: [
            '--preparation-dir', str(root/'shoe_preparation'/name),
            '--supr-model', str(FOOT_MODEL), '--output-dir', str(root/'support_fit'/name)])
        execute_parallel('run_cavity_analysis.py', lambda name: [
            '--preparation-dir', str(root/'shoe_preparation'/name),
            '--support-fit-dir', str(root/'support_fit'/name),
            '--output-dir', str(root/'cavity_analysis'/name)])
    execute_parallel('run_containment_fit.py', lambda name: [
            '--preparation-dir', str(root/'shoe_preparation'/name),
            '--support-fit-dir', str(root/'support_fit'/name),
            '--cavity-analysis-dir', str(root/'cavity_analysis'/name),
            '--supr-model', str(FOOT_MODEL), '--output-dir', str(root/'containment_fit'/name)])

    execute('run_anatomical_surface.py', ['--containment-root', str(root/'containment_fit'),
        '--supr-model', str(FOOT_MODEL), '--output-root', str(root/'anatomical_surface'), *SHOES])
    execute_parallel('run_lower_leg_attachment.py', lambda name: [
        '--anatomical-surface-root', str(root/'anatomical_surface'),
        '--preparation-root', str(root/'shoe_preparation'), '--support-fit-root', str(root/'support_fit'),
        '--full-body-supr-model', str(BODY_MODEL), '--output-root', str(root/'lower_leg_attachment'), name])
    execute('run_extended_anatomical_surface.py', ['--anatomical-surface-root', str(root/'anatomical_surface'),
        '--lower-leg-root', str(root/'lower_leg_attachment'), '--full-body-supr-model', str(BODY_MODEL),
        '--output-root', str(root/'extended_anatomical_surface'), *SHOES])

    # The canonical tetrahedral reference is already validated and shared by all shoes.
    # Copy it into this run, preserving the original and avoiding a second remeshing.
    volume_root = root / 'anatomical_volume'
    shutil.copytree(STABLE_ROOT / 'anatomical_volume/reference', volume_root / 'reference')
    execute_parallel('run_instance_anatomical_volume.py', lambda name: [
        '--anatomical-volume-root', str(volume_root),
        '--extended-anatomical-surface-root', str(root/'extended_anatomical_surface'), name])
    b3_root = root / 'instance_anatomical_volume/batch'
    execute('run_instance_volume_batch.py', ['--anatomical-volume-root', str(volume_root),
        '--extended-anatomical-surface-root', str(root/'extended_anatomical_surface'),
        '--output-root', str(b3_root), '--jobs', str(args.jobs), *SHOES])
    execute('run_instance_volume_mapping.py', ['--anatomical-volume-root', str(volume_root),
        '--instance-volume-batch-root', str(b3_root), '--containment-fit-root', str(root/'containment_fit'),
        '--output-root', str(root/'instance_volume_mapping/batch'), '--jobs', str(args.jobs), *SHOES])
    execute('run_anatomical_fibers.py', ['fibers', '--anatomical-volume-root', str(volume_root),
        '--scalar-field-root', str(STABLE_ROOT/'anatomical_fibers/converged/reference'),
        '--extended-anatomical-surface-root', str(root/'extended_anatomical_surface'),
        '--instance-volume-batch-root', str(b3_root),
        '--containment-fit-root', str(root/'containment_fit'),
        '--output-root', str(root/'anatomical_fibers/audit'),
        '--jobs', str(min(args.jobs, 8)), '--shoes', *SHOES])
    _write_json(root / 'pipeline_status.json', dict(status='coverage_review_required',
        completed_stages=stages, shoes=list(SHOES)))
    print(f'[done] 11-D coverage review: {root}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'continue'))
    parser.add_argument('--run-root', required=True, type=Path)
    parser.add_argument('--gpus', default='1,2,3,4,5')
    parser.add_argument('--jobs', type=int, default=8)
    parser.add_argument('--resume-containment', action='store_true')
    args = parser.parse_args()
    args.run_root = args.run_root.resolve()
    if not 1 <= args.jobs <= 16:
        parser.error('--jobs must be in [1, 16]')
    if args.resume_containment and args.phase != 'continue':
        parser.error('--resume-containment requires the continue phase')
    if args.phase == 'prepare':
        prepare(args)
    else:
        continue_pipeline(args)


if __name__ == '__main__':
    main()
