"""LamaInpainter 測試"""

import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock
from pathlib import Path

from watermark_remover.lama_inpainter import LamaInpainter


class TestLamaInpainter:
    """LamaInpainter 測試類"""

    def test_detect_device_mps_when_available(self):
        """測試 MPS 可用時選擇 MPS"""
        with patch("torch.backends.mps.is_available", return_value=True):
            device = LamaInpainter._detect_device_static("auto")
            assert device.type == "mps"

    def test_detect_device_cpu_when_mps_unavailable(self):
        """測試 MPS 不可用時選擇 CPU"""
        with patch("torch.backends.mps.is_available", return_value=False):
            device = LamaInpainter._detect_device_static("auto")
            assert device.type == "cpu"

    def test_detect_device_explicit_cpu(self):
        """測試明確指定 CPU"""
        device = LamaInpainter._detect_device_static("cpu")
        assert device.type == "cpu"

    def test_preprocess_image_shape(self):
        """測試圖像預處理輸出形狀"""
        # 建立測試圖像 (H, W, C) BGR
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        tensor = LamaInpainter._preprocess_image_static(image)

        # 應該是 (1, 3, H, W)
        assert tensor.shape == (1, 3, 256, 256)
        assert tensor.dtype == torch.float32

    def test_preprocess_mask_shape(self):
        """測試遮罩預處理輸出形狀"""
        # 建立測試遮罩 (H, W)
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[100:150, 100:150] = 255

        tensor = LamaInpainter._preprocess_mask_static(mask)

        # 應該是 (1, 1, H, W)
        assert tensor.shape == (1, 1, 256, 256)
        assert tensor.dtype == torch.float32

    def test_postprocess_output_shape(self):
        """測試後處理輸出形狀"""
        # 模擬模型輸出 (1, 3, H, W)
        output = torch.rand(1, 3, 256, 256)

        result = LamaInpainter._postprocess_output_static(output)

        # 應該是 (H, W, C) BGR
        assert result.shape == (256, 256, 3)
        assert result.dtype == np.uint8
