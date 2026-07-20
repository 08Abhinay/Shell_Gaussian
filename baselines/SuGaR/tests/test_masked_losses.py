from __future__ import annotations

import unittest

import torch

from sugar_utils.loss_utils import masked_l1_loss, masked_l2_loss, masked_ssim


class MaskedLossTests(unittest.TestCase):
    def test_background_pixels_do_not_contribute(self) -> None:
        prediction = torch.zeros((1, 3, 4, 4))
        target = torch.zeros_like(prediction)
        prediction[:, :, 0, 0] = 1.0
        mask = torch.zeros((1, 1, 4, 4))
        mask[:, :, 1:, 1:] = 1.0
        self.assertEqual(masked_l1_loss(prediction, target, mask).item(), 0.0)
        self.assertEqual(masked_l2_loss(prediction, target, mask).item(), 0.0)

    def test_foreground_pixels_contribute(self) -> None:
        prediction = torch.zeros((1, 3, 4, 4))
        target = torch.zeros_like(prediction)
        prediction[:, :, 2, 2] = 1.0
        mask = torch.zeros((1, 1, 4, 4))
        mask[:, :, 2, 2] = 1.0
        self.assertAlmostEqual(masked_l1_loss(prediction, target, mask).item(), 1.0)
        self.assertAlmostEqual(masked_l2_loss(prediction, target, mask).item(), 1.0)

    def test_identical_images_have_unit_masked_ssim(self) -> None:
        image = torch.rand((1, 3, 16, 16))
        mask = torch.zeros((1, 1, 16, 16))
        mask[:, :, 4:12, 4:12] = 1.0
        self.assertAlmostEqual(masked_ssim(image, image, mask).item(), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
