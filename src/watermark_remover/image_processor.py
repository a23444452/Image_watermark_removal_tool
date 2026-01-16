"""圖像處理模組：負責浮水印檢測和移除"""

import cv2
import numpy as np
from pathlib import Path


class ImageProcessor:
    """圖像處理器，用於移除浮水印"""

    def __init__(self):
        self.image: np.ndarray | None = None
        self.original: np.ndarray | None = None

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

    def remove_watermark(self, mask: np.ndarray) -> np.ndarray:
        """
        使用修復算法移除浮水印

        Args:
            mask: 二值化遮罩，白色區域為浮水印位置

        Returns:
            處理後的圖像
        """
        if self.image is None:
            raise ValueError("未載入圖像")

        # 使用 Telea 修復算法
        result = cv2.inpaint(self.image, mask, 3, cv2.INPAINT_TELEA)
        return result

    def remove_watermark_by_region(
        self, x: int, y: int, width: int, height: int
    ) -> np.ndarray:
        """
        根據指定區域移除浮水印

        Args:
            x: 區域左上角 x 座標
            y: 區域左上角 y 座標
            width: 區域寬度
            height: 區域高度

        Returns:
            處理後的圖像
        """
        if self.image is None:
            raise ValueError("未載入圖像")

        # 建立遮罩
        mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        mask[y : y + height, x : x + width] = 255

        return self.remove_watermark(mask)

    def auto_detect_watermark(self) -> np.ndarray | None:
        """
        自動檢測浮水印（檢測圖像邊緣的文字或標誌）

        Returns:
            檢測到的浮水印遮罩，如果未檢測到則返回 None
        """
        if self.image is None:
            return None

        # 轉換為灰階
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # 使用自適應閾值檢測文字
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        # 形態學操作以連接文字
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=2)

        # 在圖像四個角落檢測（常見的浮水印位置）
        h, w = thresh.shape
        border = 100  # 檢查邊界範圍

        # 檢查四個角落
        corners = [
            thresh[0:border, 0:border],  # 左上
            thresh[0:border, w - border : w],  # 右上
            thresh[h - border : h, 0:border],  # 左下
            thresh[h - border : h, w - border : w],  # 右下
        ]

        # 找到最多白色像素的角落
        max_pixels = 0
        best_corner = None
        best_position = None

        positions = [(0, 0), (0, w - border), (h - border, 0), (h - border, w - border)]

        for i, corner in enumerate(corners):
            white_pixels = cv2.countNonZero(corner)
            if white_pixels > max_pixels and white_pixels > 100:  # 閾值
                max_pixels = white_pixels
                best_corner = corner
                best_position = positions[i]

        if best_corner is None:
            return None

        # 建立完整遮罩
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
        """取得圖像尺寸（高度，寬度）"""
        if self.image is None:
            return None
        return self.image.shape[:2]

    def get_display_image(self, max_width: int = 800) -> np.ndarray | None:
        """
        取得用於顯示的圖像（調整大小以適應視窗）

        Args:
            max_width: 最大寬度

        Returns:
            調整大小後的圖像
        """
        if self.image is None:
            return None

        h, w = self.image.shape[:2]
        if w > max_width:
            scale = max_width / w
            new_w = max_width
            new_h = int(h * scale)
            return cv2.resize(self.image, (new_w, new_h))

        return self.image.copy()
