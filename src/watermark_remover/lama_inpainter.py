"""LaMa 模型推論器"""

from pathlib import Path
from typing import Callable
import numpy as np

import torch
import torch.nn.functional as F
import cv2

from .exceptions import LamaModelNotFoundError, LamaInferenceError, LamaMemoryError


class LamaInpainter:
    """LaMa 模型推論器"""

    def __init__(self, model_path: Path, device: str = "auto"):
        """
        初始化推論器

        Args:
            model_path: 模型檔案路徑
            device: 運算設備 ("auto", "mps", "cpu")
        """
        self.model = None
        self.device = self._detect_device_static(device)
        self._load_model(model_path)

    @staticmethod
    def _detect_device_static(device: str) -> torch.device:
        """偵測最佳運算設備"""
        if device == "auto":
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)

    def _load_model(self, model_path: Path) -> None:
        """載入模型"""
        if not model_path.exists():
            raise LamaModelNotFoundError(f"模型檔案不存在: {model_path}")

        try:
            # 載入模型
            self.model = torch.jit.load(str(model_path), map_location=self.device)
            self.model.eval()
        except Exception as e:
            raise LamaInferenceError(f"載入模型失敗: {e}") from e

    @staticmethod
    def _preprocess_image_static(image: np.ndarray) -> torch.Tensor:
        """
        預處理圖像

        Args:
            image: BGR 圖像 (H, W, 3)

        Returns:
            Tensor (1, 3, H, W) 範圍 [0, 1]
        """
        # BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        image_float = image_rgb.astype(np.float32) / 255.0

        # HWC to CHW
        image_chw = np.transpose(image_float, (2, 0, 1))

        # Add batch dimension
        tensor = torch.from_numpy(image_chw).unsqueeze(0)

        return tensor

    @staticmethod
    def _preprocess_mask_static(mask: np.ndarray) -> torch.Tensor:
        """
        預處理遮罩

        Args:
            mask: 二值遮罩 (H, W)，255 表示要修復的區域

        Returns:
            Tensor (1, 1, H, W) 範圍 [0, 1]
        """
        # Normalize to [0, 1]
        mask_float = mask.astype(np.float32) / 255.0

        # Add channel and batch dimensions
        tensor = torch.from_numpy(mask_float).unsqueeze(0).unsqueeze(0)

        return tensor

    @staticmethod
    def _postprocess_output_static(output: torch.Tensor) -> np.ndarray:
        """
        後處理模型輸出

        Args:
            output: Tensor (1, 3, H, W) 範圍 [0, 1]

        Returns:
            BGR 圖像 (H, W, 3) uint8
        """
        # Remove batch dimension and move to CPU
        output = output.squeeze(0).cpu()

        # Clamp to [0, 1]
        output = torch.clamp(output, 0, 1)

        # CHW to HWC
        output_hwc = output.permute(1, 2, 0).numpy()

        # Scale to [0, 255]
        output_uint8 = (output_hwc * 255).astype(np.uint8)

        # RGB to BGR
        output_bgr = cv2.cvtColor(output_uint8, cv2.COLOR_RGB2BGR)

        return output_bgr

    def _pad_to_multiple(
        self, tensor: torch.Tensor, multiple: int = 8
    ) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
        """
        填充張量使其尺寸為 multiple 的倍數

        Returns:
            (填充後的張量, (pad_left, pad_right, pad_top, pad_bottom))
        """
        _, _, h, w = tensor.shape

        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        if pad_h > 0 or pad_w > 0:
            tensor = F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')

        return tensor, (pad_left, pad_right, pad_top, pad_bottom)

    def _unpad(
        self, tensor: torch.Tensor, padding: tuple[int, int, int, int]
    ) -> torch.Tensor:
        """移除填充"""
        pad_left, pad_right, pad_top, pad_bottom = padding
        _, _, h, w = tensor.shape

        return tensor[
            :, :,
            pad_top : h - pad_bottom if pad_bottom > 0 else h,
            pad_left : w - pad_right if pad_right > 0 else w
        ]

    def inpaint(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        progress_callback: Callable[[int], None] | None = None,
    ) -> np.ndarray:
        """
        執行圖像修復

        Args:
            image: BGR 圖像 (H, W, 3)
            mask: 二值遮罩 (H, W)，255 表示要修復的區域
            progress_callback: 進度回調 (0-100)

        Returns:
            修復後的 BGR 圖像
        """
        if self.model is None:
            raise LamaInferenceError("模型尚未載入")

        try:
            if progress_callback:
                progress_callback(10)

            # 預處理
            image_tensor = self._preprocess_image_static(image).to(self.device)
            mask_tensor = self._preprocess_mask_static(mask).to(self.device)

            if progress_callback:
                progress_callback(30)

            # 填充至 8 的倍數
            image_tensor, padding = self._pad_to_multiple(image_tensor)
            mask_tensor, _ = self._pad_to_multiple(mask_tensor)

            if progress_callback:
                progress_callback(40)

            # 推論
            with torch.no_grad():
                output = self.model(image_tensor, mask_tensor)

            if progress_callback:
                progress_callback(80)

            # 移除填充
            output = self._unpad(output, padding)

            # 後處理
            result = self._postprocess_output_static(output)

            if progress_callback:
                progress_callback(100)

            return result

        except torch.cuda.OutOfMemoryError:
            raise LamaMemoryError("GPU 記憶體不足")
        except Exception as e:
            raise LamaInferenceError(f"推論過程錯誤: {e}") from e

    def unload_model(self) -> None:
        """釋放模型記憶體"""
        if self.model is not None:
            del self.model
            self.model = None

            # 清理記憶體
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
