"""Retroactively compute SSIM and LPIPS from saved validation image pairs and update metrics.txt."""
import os
import sys
import glob
import numpy as np
import torch
import imageio
from pytorch_msssim import ssim as compute_ssim_fn
import lpips

def main():
    validate_dir = sys.argv[1] if len(sys.argv) > 1 else \
        "output/Adidas-Yeezy-Boost-350-V2-Static-Non-Reflective-Kids_normfix/validate"

    opt_paths = sorted(glob.glob(os.path.join(validate_dir, "val_*_opt.png")))
    ref_paths = sorted(glob.glob(os.path.join(validate_dir, "val_*_ref.png")))

    assert len(opt_paths) == len(ref_paths), f"Mismatch: {len(opt_paths)} opt vs {len(ref_paths)} ref"
    assert len(opt_paths) > 0, f"No image pairs found in {validate_dir}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_model = lpips.LPIPS(net='vgg').to(device)

    mse_values, psnr_values, ssim_values, lpips_values = [], [], [], []

    print(f"Computing metrics for {len(opt_paths)} image pairs in {validate_dir}")
    for i, (opt_path, ref_path) in enumerate(zip(opt_paths, ref_paths)):
        opt_np = imageio.imread(opt_path).astype(np.float32) / 255.0
        ref_np = imageio.imread(ref_path).astype(np.float32) / 255.0

        opt_t = torch.from_numpy(opt_np).to(device)
        ref_t = torch.from_numpy(ref_np).to(device)

        # MSE & PSNR
        mse = torch.nn.functional.mse_loss(opt_t, ref_t).item()
        psnr = -10.0 / np.log(10.0) * np.log(mse)

        # SSIM (needs NCHW)
        opt_nchw = opt_t.unsqueeze(0).permute(0, 3, 1, 2)
        ref_nchw = ref_t.unsqueeze(0).permute(0, 3, 1, 2)
        ssim_val = compute_ssim_fn(opt_nchw, ref_nchw, data_range=1.0, size_average=True).item()

        # LPIPS (needs NCHW in [-1, 1])
        lpips_val = lpips_model(opt_nchw * 2 - 1, ref_nchw * 2 - 1).item()

        mse_values.append(mse)
        psnr_values.append(psnr)
        ssim_values.append(ssim_val)
        lpips_values.append(lpips_val)

        print(f"  [{i:3d}] MSE={mse:.8f}  PSNR={psnr:.3f}  SSIM={ssim_val:.6f}  LPIPS={lpips_val:.6f}")

    avg_mse = np.mean(mse_values)
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    avg_lpips = np.mean(lpips_values)

    print(f"\nAVERAGES: MSE={avg_mse:.4f}  PSNR={avg_psnr:.3f}  SSIM={avg_ssim:.4f}  LPIPS={avg_lpips:.4f}")

    metrics_path = os.path.join(validate_dir, "metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write("ID, MSE, PSNR, SSIM, LPIPS\n")
        for i in range(len(mse_values)):
            f.write("%d, %1.8f, %1.8f, %1.8f, %1.8f\n" % (
                i, mse_values[i], psnr_values[i], ssim_values[i], lpips_values[i]))
        f.write("AVERAGES: %1.4f, %2.3f, %1.4f, %1.4f\n" % (avg_mse, avg_psnr, avg_ssim, avg_lpips))

    print(f"Updated {metrics_path}")

if __name__ == "__main__":
    main()
