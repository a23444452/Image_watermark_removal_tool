"""ImageProcessor LaMa 整合測試"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from watermark_remover.image_processor import ImageProcessor, InpaintMethod


class TestImageProcessorLama:
    """ImageProcessor LaMa 整合測試類"""

    def test_inpaint_method_enum_has_lama(self):
        """測試 InpaintMethod 包含 LAMA"""
        assert hasattr(InpaintMethod, "LAMA")
        assert InpaintMethod.LAMA.value == "lama"

    def test_default_method_is_lama(self):
        """測試預設方法為 LAMA"""
        processor = ImageProcessor()
        assert processor.default_method == InpaintMethod.LAMA

    def test_remove_watermark_lama_fallback_to_ns(self):
        """測試 LaMa 不可用時降級到 NS"""
        processor = ImageProcessor()

        # 建立測試圖像和遮罩
        processor.image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 255

        # Mock LaMa 不可用
        with patch.object(processor, '_lama_available', return_value=False):
            result = processor.remove_watermark(mask, method=InpaintMethod.LAMA)

        # 應該成功返回結果（使用 NS 後備）
        assert result is not None
        assert result.shape == processor.image.shape
