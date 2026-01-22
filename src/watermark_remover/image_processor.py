"""圖像處理模組：負責浮水印檢測和移除"""

import cv2
import numpy as np
from pathlib import Path
from enum import Enum
from typing import Callable

from .exceptions import LamaError


class InpaintMethod(Enum):
    """修復算法類型"""

    LAMA = "lama"  # LaMa AI 模型（預設）
    TELEA = "telea"  # 快速行進法 (Fast Marching Method)
    NS = "ns"  # Navier-Stokes 流體力學法
    HYBRID = "hybrid"  # 混合法（結合兩種算法）


class ImageProcessor:
    """圖像處理器，用於移除浮水印"""

    def __init__(self):
        self.image: np.ndarray | None = None
        self.original: np.ndarray | None = None
        self.default_method = InpaintMethod.LAMA  # 預設使用 LaMa
        self.default_radius = 5

        # LaMa 相關
        self._lama_inpainter = None
        self._lama_checked = False

    def _lama_available(self) -> bool:
        """檢查 LaMa 是否可用"""
        if self._lama_checked:
            return self._lama_inpainter is not None

        self._lama_checked = True

        try:
            from .model_manager import ModelManager
            from .lama_inpainter import LamaInpainter

            manager = ModelManager()
            if manager.is_model_downloaded("lama"):
                model_path = manager.get_model_path("lama")
                self._lama_inpainter = LamaInpainter(model_path)
                return True
        except Exception:
            pass

        return False

    def _get_lama_inpainter(self):
        """取得 LaMa 推論器"""
        if not self._lama_available():
            return None
        return self._lama_inpainter

    def load_image(self, image_path: str | Path) -> bool:
        """載入圖像"""
        try:
            self.original = cv2.imread(str(image_path))
            if self.original is None:
                return False
            self.image = self.original.copy()
            return True
        except Exception:
            return False

    def remove_watermark(
        self,
        mask: np.ndarray,
        method: InpaintMethod = None,
        radius: int = None,
        progress_callback: Callable[[int], None] | None = None,
        fallback_callback: Callable[[str], None] | None = None,
    ) -> np.ndarray:
        """
        使用修復算法移除浮水印

        Args:
            mask: 二值化遮罩，白色區域為浮水印位置
            method: 修復算法類型
            radius: 修復半徑（像素），範圍 1-10，值越大修復範圍越廣
            progress_callback: 進度回調 (0-100)
            fallback_callback: 後備算法通知回調

        Returns:
            處理後的圖像
        """
        if self.image is None:
            raise ValueError("未載入圖像")

        # 使用預設值
        if method is None:
            method = self.default_method
        if radius is None:
            radius = self.default_radius

        # 嘗試使用 LaMa
        if method == InpaintMethod.LAMA:
            lama = self._get_lama_inpainter()
            if lama is not None:
                try:
                    return lama.inpaint(self.image, mask, progress_callback)
                except LamaError:
                    pass

            # LaMa 不可用，降級到 NS
            if fallback_callback:
                fallback_callback("LaMa 模型不可用，已切換至 NS 算法")
            method = InpaintMethod.NS

        # 擴展遮罩以包含更多周圍區域
        kernel = np.ones((3, 3), np.uint8)
        expanded_mask = cv2.dilate(mask, kernel, iterations=1)

        if progress_callback:
            progress_callback(30)

        # 根據選擇的方法進行修復
        if method == InpaintMethod.TELEA:
            result = cv2.inpaint(self.image, expanded_mask, radius, cv2.INPAINT_TELEA)

        elif method == InpaintMethod.NS:
            result = cv2.inpaint(self.image, expanded_mask, radius, cv2.INPAINT_NS)

        elif method == InpaintMethod.HYBRID:
            result = cv2.inpaint(self.image, expanded_mask, radius, cv2.INPAINT_NS)
            result = cv2.inpaint(
                result, expanded_mask, max(radius - 2, 1), cv2.INPAINT_TELEA
            )
        else:
            result = cv2.inpaint(self.image, expanded_mask, radius, cv2.INPAINT_NS)

        if progress_callback:
            progress_callback(80)

        # 後處理
        result = cv2.bilateralFilter(result, 5, 50, 50)

        if progress_callback:
            progress_callback(100)

        return result

    def remove_watermark_by_region(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        method: InpaintMethod = None,
        radius: int = None,
        progress_callback: Callable[[int], None] | None = None,
        fallback_callback: Callable[[str], None] | None = None,
    ) -> np.ndarray:
        """根據指定區域移除浮水印"""
        if self.image is None:
            raise ValueError("未載入圖像")

        mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        mask[y : y + height, x : x + width] = 255

        return self.remove_watermark(mask, method, radius, progress_callback, fallback_callback)

    def remove_watermark_advanced(
        self, mask: np.ndarray, radius: int = 7
    ) -> np.ndarray:
        """進階浮水印移除（使用多層次處理）"""
        if self.image is None:
            raise ValueError("未載入圖像")

        kernel = np.ones((3, 3), np.uint8)
        expanded_mask = cv2.dilate(mask, kernel, iterations=2)

        result = cv2.inpaint(self.image, expanded_mask, radius, cv2.INPAINT_NS)

        repair_region = cv2.bitwise_and(result, result, mask=expanded_mask)
        repair_region = cv2.fastNlMeansDenoisingColored(
            repair_region, None, 10, 10, 7, 21
        )

        result = cv2.bitwise_and(result, result, mask=cv2.bitwise_not(expanded_mask))
        result = cv2.add(result, repair_region)

        edge_mask = cv2.dilate(expanded_mask, kernel, iterations=1) - expanded_mask
        blurred = cv2.GaussianBlur(result, (5, 5), 0)
        result = np.where(edge_mask[:, :, np.newaxis] > 0, blurred, result)

        result = cv2.bilateralFilter(result, 5, 75, 75)

        return result

    def auto_detect_watermark(self) -> np.ndarray | None:
        """自動檢測浮水印"""
        if self.image is None:
            return None

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=2)

        h, w = thresh.shape
        border = 100

        corners = [
            thresh[0:border, 0:border],
            thresh[0:border, w - border : w],
            thresh[h - border : h, 0:border],
            thresh[h - border : h, w - border : w],
        ]

        max_pixels = 0
        best_corner = None
        best_position = None

        positions = [(0, 0), (0, w - border), (h - border, 0), (h - border, w - border)]

        for i, corner in enumerate(corners):
            white_pixels = cv2.countNonZero(corner)
            if white_pixels > max_pixels and white_pixels > 100:
                max_pixels = white_pixels
                best_corner = corner
                best_position = positions[i]

        if best_corner is None:
            return None

        mask = np.zeros((h, w), dtype=np.uint8)
        y, x = best_position
        mask[y : y + border, x : x + border] = best_corner

        return mask

    def save_image(self, image: np.ndarray, output_path: str | Path) -> bool:
        """儲存圖像"""
        try:
            cv2.imwrite(str(output_path), image)
            return True
        except Exception:
            return False

    def get_image_shape(self) -> tuple[int, int] | None:
        """取得圖像尺寸"""
        if self.image is None:
            return None
        return self.image.shape[:2]

    def get_display_image(self, max_width: int = 800) -> np.ndarray | None:
        """取得用於顯示的圖像"""
        if self.image is None:
            return None

        h, w = self.image.shape[:2]
        if w > max_width:
            scale = max_width / w
            new_w = max_width
            new_h = int(h * scale)
            return cv2.resize(self.image, (new_w, new_h))

        return self.image.copy()

    def compare_methods(
        self, mask: np.ndarray, radius: int = 5
    ) -> dict[str, np.ndarray]:
        """比較不同修復算法的效果"""
        if self.image is None:
            raise ValueError("未載入圖像")

        results = {}

        # LaMa（如果可用）
        if self._lama_available():
            try:
                results["LaMa (AI)"] = self.remove_watermark(mask, InpaintMethod.LAMA, radius)
            except Exception:
                pass

        results["Telea"] = self.remove_watermark(mask, InpaintMethod.TELEA, radius)
        results["NS"] = self.remove_watermark(mask, InpaintMethod.NS, radius)
        results["混合算法"] = self.remove_watermark(mask, InpaintMethod.HYBRID, radius)
        results["進階處理"] = self.remove_watermark_advanced(mask, radius)

        return results
