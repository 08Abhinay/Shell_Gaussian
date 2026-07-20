import argparse
import os
import subprocess
import sys
from sugar_utils.general_utils import str2bool


class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.__dict__ = self


if __name__ == "__main__":
    # ----- Parser -----
    parser = argparse.ArgumentParser(description='Script to optimize a full SuGaR model.')
    
    # Data and vanilla 3DGS checkpoint
    parser.add_argument('-s', '--scene_path',
                        type=str, 
                        help='(Required) path to the scene data to use.')  
    
    # Vanilla 3DGS optimization at beginning
    parser.add_argument('--gs_output_dir', type=str, default=None,
                        help='(Optional) If None, will automatically train a vanilla Gaussian Splatting model at the beginning of the training. '
                        'Else, skips the vanilla Gaussian Splatting optimization and use the checkpoint in the provided directory.')
    
    # Regularization for coarse SuGaR
    parser.add_argument('-r', '--regularization_type', type=str,
                        help='(Required) Type of regularization to use for coarse SuGaR. Can be "sdf", "density" or "dn_consistency". ' 
                        'We recommend using "dn_consistency" for the best mesh quality.')
    
    # Extract mesh
    parser.add_argument('-l', '--surface_level', type=float, default=0.3, 
                        help='Surface level to extract the mesh at. Default is 0.3')
    parser.add_argument('-v', '--n_vertices_in_mesh', type=int, default=1_000_000, 
                        help='Number of vertices in the extracted mesh.')
    parser.add_argument('--project_mesh_on_surface_points', type=str2bool, default=True, 
                        help='If True, project the mesh on the surface points for better details.')
    parser.add_argument('-b', '--bboxmin', type=str, default=None, 
                        help='Min coordinates to use for foreground.')  
    parser.add_argument('-B', '--bboxmax', type=str, default=None, 
                        help='Max coordinates to use for foreground.')
    parser.add_argument('--center_bbox', type=str2bool, default=True, 
                        help='If True, center the bbox. Default is False.')
    parser.add_argument('--foreground_only', type=str2bool, default=False,
                        help='If True, discard geometry outside the foreground bbox during mesh extraction.')
    parser.add_argument('--use_masks', type=str2bool, default=False,
                        help='Use image alpha as a foreground mask for SuGaR optimization and extraction.')
    parser.add_argument('--entropy_regularization', type=str2bool, default=True)
    parser.add_argument('--prune_at_start', type=str2bool, default=False)
    parser.add_argument('--constrain_points_to_bbox', type=str2bool, default=False)
    parser.add_argument('--max_gaussian_scale_ratio', type=float, default=None)
    parser.add_argument('--filter_gaussians_by_bbox', type=str2bool, default=False)
    parser.add_argument('--poisson_depth', type=int, default=10)
    parser.add_argument('--vertices_density_quantile', type=float, default=0.1)
    parser.add_argument('--surface_point_budget', type=int, default=10_000_000)
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Optional output namespace under output/.')
    
    # Parameters for refined SuGaR
    parser.add_argument('-g', '--gaussians_per_triangle', type=int, default=1, 
                        help='Number of gaussians per triangle.')
    parser.add_argument('-f', '--refinement_iterations', type=int, default=15_000, 
                        help='Number of refinement iterations.')
    
    # (Optional) Parameters for textured mesh extraction
    parser.add_argument('-t', '--export_obj', type=str2bool, default=True, 
                        help='If True, will export a textured mesh as an .obj file from the refined SuGaR model. '
                        'Computing a traditional colored UV texture should take less than 10 minutes.')
    parser.add_argument('--square_size',
                        default=8, type=int, help='Size of the square to use for the UV texture.')
    parser.add_argument('--postprocess_mesh', type=str2bool, default=False, 
                        help='If True, postprocess the mesh by removing border triangles with low-density. '
                        'This step takes a few minutes and is not needed in general, as it can also be risky. '
                        'However, it increases the quality of the mesh in some cases, especially when an object is visible only from one side.')
    parser.add_argument('--postprocess_density_threshold', type=float, default=0.1,
                        help='Threshold to use for postprocessing the mesh.')
    parser.add_argument('--postprocess_iterations', type=int, default=5,
                        help='Number of iterations to use for postprocessing the mesh.')
    
    # (Optional) PLY file export
    parser.add_argument('--export_ply', type=str2bool, default=True,
                        help='If True, export a ply file with the refined 3D Gaussians at the end of the training. '
                        'This file can be large (+/- 500MB), but is needed for using the dedicated viewer. Default is True.')
    
    # (Optional) Default configurations
    parser.add_argument('--low_poly', type=str2bool, default=False, 
                        help='Use standard config for a low poly mesh, with 200k vertices and 6 Gaussians per triangle.')
    parser.add_argument('--high_poly', type=str2bool, default=False,
                        help='Use standard config for a high poly mesh, with 1M vertices and 1 Gaussians per triangle.')
    parser.add_argument('--refinement_time', type=str, default=None, 
                        help="Default configs for time to spend on refinement. Can be 'short', 'medium' or 'long'.")
      
    # Evaluation split
    parser.add_argument('--eval', type=str2bool, default=True, help='Use eval split.')

    # GPU
    parser.add_argument('--gpu', type=int, default=0, help='Index of GPU device to use.')
    parser.add_argument('--white_background', type=str2bool, default=False, help='Use a white background instead of black.')

    # Parse arguments
    args = parser.parse_args()
    if args.low_poly:
        args.n_vertices_in_mesh = 200_000
        args.gaussians_per_triangle = 6
        print('Using low poly config.')
    if args.high_poly:
        args.n_vertices_in_mesh = 1_000_000
        args.gaussians_per_triangle = 1
        print('Using high poly config.')
    if args.refinement_time == 'short':
        args.refinement_iterations = 2_000
        print('Using short refinement time.')
    if args.refinement_time == 'medium':
        args.refinement_iterations = 7_000
        print('Using medium refinement time.')
    if args.refinement_time == 'long':
        args.refinement_iterations = 15_000
        print('Using long refinement time.')
    if args.export_obj:
        print('Will export a UV-textured mesh as an .obj file.')
    if args.export_ply:
        print('Will export a ply file with the refined 3D Gaussians at the end of the training.')
        
    # Output directory for the vanilla 3DGS checkpoint
    if args.gs_output_dir is None:
        sep = os.path.sep
        scene_name = args.scene_path.rstrip(sep).split(sep)[-1]
        if args.experiment_name:
            gs_checkpoint_dir = os.path.join("output", args.experiment_name, "vanilla_gs", scene_name)
        elif len(args.scene_path.split(sep)[-1]) > 0:
            gs_checkpoint_dir = os.path.join("output", "vanilla_gs", args.scene_path.split(sep)[-1])
        else:
            gs_checkpoint_dir = os.path.join("output", "vanilla_gs", args.scene_path.split(sep)[-2])
        gs_checkpoint_dir = gs_checkpoint_dir + sep

        # Trains a 3DGS scene for 7k iterations
        gs_command = [
            sys.executable,
            "./gaussian_splatting/train.py",
            "-s", args.scene_path,
            "-m", gs_checkpoint_dir,
            "--iterations", "7000",
        ]
        if args.white_background:
            gs_command.append("-w")
        subprocess.run(
            gs_command,
            check=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu)},
        )
    else:
        print("A vanilla 3DGS checkpoint was provided. Skipping the vanilla 3DGS optimization.")
        gs_checkpoint_dir = args.gs_output_dir
        if gs_checkpoint_dir[-1] != os.path.sep:
            gs_checkpoint_dir += os.path.sep
    
    # Use a checked argv list so child failures propagate to tmux and paths do
    # not depend on shell quoting.
    sugar_command = [
            sys.executable,
            "train.py",
            "-s", args.scene_path,
            "-c", gs_checkpoint_dir,
            "-i", "7000",
            "-r", args.regularization_type,
            "-l", str(args.surface_level),
            "-v", str(args.n_vertices_in_mesh),
            "--project_mesh_on_surface_points", str(args.project_mesh_on_surface_points),
            "-g", str(args.gaussians_per_triangle),
            "-f", str(args.refinement_iterations),
            f"--bboxmin={args.bboxmin}",
            f"--bboxmax={args.bboxmax}",
            "--center_bbox", str(args.center_bbox),
            "--foreground_only", str(args.foreground_only),
            "--use_masks", str(args.use_masks),
            "--entropy_regularization", str(args.entropy_regularization),
            "--prune_at_start", str(args.prune_at_start),
            "--constrain_points_to_bbox", str(args.constrain_points_to_bbox),
            "--filter_gaussians_by_bbox", str(args.filter_gaussians_by_bbox),
            "--poisson_depth", str(args.poisson_depth),
            "--vertices_density_quantile", str(args.vertices_density_quantile),
            "--surface_point_budget", str(args.surface_point_budget),
            "-t", str(args.export_obj),
            "--square_size", str(args.square_size),
            "--postprocess_mesh", str(args.postprocess_mesh),
            "--postprocess_density_threshold", str(args.postprocess_density_threshold),
            "--postprocess_iterations", str(args.postprocess_iterations),
            "--export_ply", str(args.export_ply),
            "--low_poly", str(args.low_poly),
            "--high_poly", str(args.high_poly),
            "--refinement_time", str(args.refinement_time),
            "--eval", str(args.eval),
            "--gpu", str(args.gpu),
            "--white_background", str(args.white_background),
        ]
    if args.max_gaussian_scale_ratio is not None:
        sugar_command.extend(
            ["--max_gaussian_scale_ratio", str(args.max_gaussian_scale_ratio)]
        )
    if args.experiment_name is not None:
        sugar_command.extend(["--experiment_name", args.experiment_name])
    subprocess.run(
        sugar_command,
        check=True,
    )
