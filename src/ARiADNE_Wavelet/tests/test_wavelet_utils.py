import unittest

import torch

from ARiADNE_Wavelet.wavelet import haar_decompose_2d, haar_decompose_last_dim, haar_decompose_time, haar_decompose_vector


class WaveletUtilsTests(unittest.TestCase):
    def test_haar_decompose_vector_odd_length(self):
        x = torch.arange(7, dtype=torch.float32)
        low, highs = haar_decompose_vector(x, levels=2)
        self.assertGreaterEqual(low.numel(), 1)
        self.assertEqual(len(highs), 2)
        self.assertTrue(torch.isfinite(low).all())
        self.assertTrue(torch.isfinite(highs[0]).all())

    def test_haar_decompose_time_shape(self):
        x = torch.randn(3, 9, 4)
        low, highs = haar_decompose_time(x, levels=2)
        self.assertEqual(low.dim(), 3)
        self.assertEqual(low.size(0), 3)
        self.assertEqual(low.size(2), 4)
        self.assertEqual(len(highs), 2)

    def test_haar_decompose_last_dim_shape(self):
        x = torch.randn(2, 5, 7)
        low, highs = haar_decompose_last_dim(x, levels=2)
        self.assertEqual(low.size(0), 2)
        self.assertEqual(low.size(1), 5)
        self.assertEqual(len(highs), 2)

    def test_haar_decompose_2d_shape(self):
        x = torch.randn(3, 5, 7)
        low, highs = haar_decompose_2d(x, levels=2)
        self.assertEqual(low.dim(), 3)
        self.assertEqual(low.size(0), 3)
        self.assertEqual(len(highs), 2)
        for bands in highs:
            self.assertEqual(len(bands), 3)


if __name__ == "__main__":
    unittest.main()
