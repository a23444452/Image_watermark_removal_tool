"""圖像處理模組：負責浮水印檢測和移除"""

import cv2
import numpy as np
from pathlib import Path
from enum import Enum


class InpaintMethod(Enum):
    """修復算法類型"""

    TELEA = "telea"  # 快速行進法 (Fast Marching Method)
    NS = "ns"  # Navier-Stokes 流體力學法
    HYBRID = "hybrid"  # 混合法（結合兩種算法）


class ImageProcessor:
    """圖像處理器，用於移除浮水印"""

    def __init__(self):
        self.image: np.ndarray | None = None
        self.original: np.ndarray | None = None
        self.default_method = InpaintMethod.NS  # 預設使用 NS 算法
        self.default_radius = 5  # 增加修復半徑

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
    ) -> np.ndarray:
        """
        使用修復算法移除浮水印

        Args:
            mask: 二值化遮罩，白色區域為浮水印位置
            method: 修復算法類型
            radius: 修復半徑（像素），範圍 1-10，值越大修復範圍越廣

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

        # 擴展遮罩以包含更多周圍區域
        kernel = np.ones((3, 3), np.uint8)
        expanded_mask = cv2.dilate(mask, kernel, iterations=1)

        # 根據選擇的方法進行修復
        if method == InpaintMethod.TELEA:
            # Telea 算法：快速，適合小面積
            result = cv2.inpaint(self.image, expanded_mask, radius, cv2.INPAINT_TELEA)

        elif method == InpaintMethod.NS:
            # Navier-Stokes 算法：品質更好，適合較大面積
            result = cv2.inpaint(self.image, expanded_mask, radius, cv2.INPAINT_NS)

        elif method == InpaintMethod.HYBRID:
            # 混合方法：先用 NS 再用 Telea
            # 第一遍：NS 算法處理主要區域
            result = cv2.inpaint(self.image, expanded_mask, radius, cv2.INPAINT_NS)

            # 第二遍：Telea 算法平滑邊緣
            result = cv2.inpaint(
                result, expanded_mask, max(radius - 2, 1), cv2.INPAINT_TELEA
            )
        else:
            result = cv2.inpaint(self.image, expanded_mask, radius, cv2.INPAINT_NS)

        # 後處理：使用雙邊濾波器平滑結果，保留邊緣
        result = cv2.bilateralFilter(result, 5, 50, 50)

        return result

    def remove_watermark_by_region(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        method: InpaintMethod = None,
        radius: int = None,
    ) -> np.ndarray:
        """
        根據指定區域移除浮水印

        Args:
            x: 區域左上角 x 座標
            y: 區域左上角 y 座標
            width: 區域寬度
            height: 區域高度
            method: 修復算法類型
            radius: 修復半徑

        Returns:
            處理後的圖像
        """
        if self.image is None:
            raise ValueError("未載入圖像")

        # 建立遮罩
        mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        mask[y : y + height, x : x + width] = 255

        return self.remove_watermark(mask, method, radius)

    def remove_watermark_advanced(
        self, mask: np.ndarray, radius: int = 7
    ) -> np.ndarray:
        """
        進階浮水印移除（使用多層次處理）

        Args:
            mask: 二值化遮罩
            radius: 修復半徑

        Returns:
            處理後的圖像
        """
        if self.image is None:
            raise ValueError("未載入圖像")

        # 1. 擴展遮罩
        kernel = np.ones((3, 3), np.uint8)
        expanded_mask = cv2.dilate(mask, kernel, iterations=2)

        # 2. 使用 NS 算法進行主要修復
        result = cv2.inpaint(self.image, expanded_mask, radius, cv2.INPAINT_NS)

        # 3. 對修復區域進行細緻處理
        # 提取修復區域
        repair_region = cv2.bitwise_and(result, result, mask=expanded_mask)

        # 使用非局部均值去噪
        repair_region = cv2.fastNlMeansDenoisingColored(
            repair_region, None, 10, 10, 7, 21
        )

        # 將處理後的區域合併回去
        result = cv2.bitwise_and(result, result, mask=cv2.bitwise_not(expanded_mask))
        result = cv2.add(result, repair_region)

        # 4. 邊緣平滑
        # 只在遮罩邊緣進行高斯模糊
        edge_mask = cv2.dilate(expanded_mask, kernel, iterations=1) - expanded_mask
        blurred = cv2.GaussianBlur(result, (5, 5), 0)
        result = np.where(edge_mask[:, :, np.newaxis] > 0, blurred, result)

        # 5. 整體平滑（保留邊緣）
        result = cv2.bilateralFilter(result, 5, 75, 75)

        return result

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

    def compare_methods(
        self, mask: np.ndarray, radius: int = 5
    ) -> dict[str, np.ndarray]:
        """
        比較不同修復算法的效果

        Args:
            mask: 浮水印遮罩
            radius: 修復半徑

        Returns:
            包含不同算法結果的字典
        """
        if self.image is None:
            raise ValueError("未載入圖像")

        results = {}

        # Telea 算法
        results["Telea"] = self.remove_watermark(
            mask, InpaintMethod.TELEA, radius
        )

        # NS 算法
        results["NS (推薦)"] = self.remove_watermark(mask, InpaintMethod.NS, radius)

        # 混合算法
        results["混合算法"] = self.remove_watermark(
            mask, InpaintMethod.HYBRID, radius
        )

        # 進階算法
        results["進階處理"] = self.remove_watermark_advanced(mask, radius)

        return results
