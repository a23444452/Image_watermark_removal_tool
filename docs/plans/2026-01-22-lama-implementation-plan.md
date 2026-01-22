# LaMa 模型整合實作計畫

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 整合 LaMa AI 模型以提升大面積浮水印移除品質

**Architecture:** 新增 ModelManager 處理模型下載/管理，LamaInpainter 處理模型推論，修改 ImageProcessor 整合 LaMa 方法，修改 GUI 新增算法選單與啟動檢查

**Tech Stack:** PyTorch 2.0+, Apple Silicon MPS, PySide6

---

## Task 1: 新增 PyTorch 依賴

**Files:**
- Modify: `pyproject.toml:5-10`

**Step 1: 修改 pyproject.toml 新增 torch 依賴**

```toml
[project]
name = "watermark-remover"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "opencv-python>=4.8.0",
    "numpy>=1.24.0",
    "Pillow>=10.0.0",
    "PySide6>=6.6.0",
    "torch>=2.0.0",
    "torchvision>=0.15.0",
]
```

**Step 2: 安裝新依賴**

Run: `pip install -e .`
Expected: 成功安裝 torch 和 torchvision

**Step 3: 驗證 PyTorch 安裝**

Run: `python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'MPS available: {torch.backends.mps.is_available()}')"`
Expected: 顯示 PyTorch 版本和 MPS 可用狀態

**Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "feat: add PyTorch dependencies for LaMa model support"
```

---

## Task 2: 建立例外類別

**Files:**
- Create: `src/watermark_remover/exceptions.py`

**Step 1: 建立例外類別檔案**

```python
"""LaMa 相關例外類別"""


class LamaError(Exception):
    """LaMa 相關錯誤基類"""
    pass


class LamaModelNotFoundError(LamaError):
    """模型檔案不存在"""
    pass


class LamaDownloadError(LamaError):
    """模型下載失敗"""
    pass


class LamaInferenceError(LamaError):
    """推論過程錯誤"""
    pass


class LamaMemoryError(LamaError):
    """記憶體不足"""
    pass
```

**Step 2: Commit**

```bash
git add src/watermark_remover/exceptions.py
git commit -m "feat: add LaMa exception classes"
```

---

## Task 3: 建立 ModelManager（測試先行）

**Files:**
- Create: `tests/test_model_manager.py`
- Create: `src/watermark_remover/model_manager.py`

**Step 1: 建立 ModelManager 測試**

```python
"""ModelManager 測試"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import os

from watermark_remover.model_manager import ModelManager


class TestModelManager:
    """ModelManager 測試類"""

    def test_model_dir_created_on_init(self):
        """測試初始化時建立模型目錄"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(ModelManager, 'MODEL_DIR', Path(tmpdir) / "models"):
                manager = ModelManager()
                assert manager.model_dir.exists()

    def test_is_model_downloaded_false_when_not_exists(self):
        """測試模型未下載時返回 False"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(ModelManager, 'MODEL_DIR', Path(tmpdir) / "models"):
                manager = ModelManager()
                assert manager.is_model_downloaded("lama") is False

    def test_is_model_downloaded_true_when_exists(self):
        """測試模型存在時返回 True"""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "models" / "lama"
            model_dir.mkdir(parents=True)
            (model_dir / "big-lama.pt").touch()

            with patch.object(ModelManager, 'MODEL_DIR', Path(tmpdir) / "models"):
                manager = ModelManager()
                assert manager.is_model_downloaded("lama") is True

    def test_get_model_path(self):
        """測試取得模型路徑"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(ModelManager, 'MODEL_DIR', Path(tmpdir) / "models"):
                manager = ModelManager()
                path = manager.get_model_path("lama")
                assert path == Path(tmpdir) / "models" / "lama" / "big-lama.pt"
```

**Step 2: 執行測試確認失敗**

Run: `pytest tests/test_model_manager.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'watermark_remover.model_manager'"

**Step 3: 實作 ModelManager**

```python
"""模型管理器：負責 AI 模型的下載和管理"""

from pathlib import Path
from typing import Callable
import urllib.request
import shutil

from .exceptions import LamaDownloadError, LamaModelNotFoundError


class ModelManager:
    """管理 AI 模型的下載與儲存"""

    MODEL_DIR = Path.home() / ".cache" / "watermark-remover" / "models"

    # LaMa 模型資訊
    MODELS = {
        "lama": {
            "filename": "big-lama.pt",
            "url": "https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.pt",
            "size": 200 * 1024 * 1024,  # 約 200 MB
        }
    }

    def __init__(self):
        self.model_dir = self.MODEL_DIR
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def is_model_downloaded(self, model_name: str = "lama") -> bool:
        """檢查模型是否已下載"""
        model_path = self.get_model_path(model_name)
        return model_path.exists()

    def get_model_path(self, model_name: str = "lama") -> Path:
        """取得模型路徑"""
        if model_name not in self.MODELS:
            raise ValueError(f"未知的模型: {model_name}")

        model_info = self.MODELS[model_name]
        return self.model_dir / model_name / model_info["filename"]

    def get_model_size(self, model_name: str = "lama") -> int:
        """取得模型預估大小（bytes）"""
        if model_name not in self.MODELS:
            raise ValueError(f"未知的模型: {model_name}")
        return self.MODELS[model_name]["size"]

    def download_model(
        self,
        model_name: str = "lama",
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> Path:
        """
        下載模型

        Args:
            model_name: 模型名稱
            progress_callback: 進度回調函數 (downloaded_bytes, total_bytes)

        Returns:
            下載完成的模型路徑

        Raises:
            LamaDownloadError: 下載失敗
        """
        if model_name not in self.MODELS:
            raise ValueError(f"未知的模型: {model_name}")

        model_info = self.MODELS[model_name]
        model_path = self.get_model_path(model_name)

        # 建立模型目錄
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # 下載模型
        try:
            temp_path = model_path.with_suffix(".tmp")

            # 使用 urllib 下載
            with urllib.request.urlopen(model_info["url"]) as response:
                total_size = int(response.headers.get("Content-Length", 0))
                downloaded = 0
                chunk_size = 8192

                with open(temp_path, "wb") as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)

                        if progress_callback:
                            progress_callback(downloaded, total_size)

            # 下載完成，重命名檔案
            shutil.move(str(temp_path), str(model_path))

            return model_path

        except Exception as e:
            # 清理臨時檔案
            temp_path = model_path.with_suffix(".tmp")
            if temp_path.exists():
                temp_path.unlink()
            raise LamaDownloadError(f"下載模型失敗: {e}") from e

    def delete_model(self, model_name: str = "lama") -> bool:
        """刪除已下載的模型"""
        model_path = self.get_model_path(model_name)
        if model_path.exists():
            model_path.unlink()
            return True
        return False
```

**Step 4: 執行測試確認通過**

Run: `pytest tests/test_model_manager.py -v`
Expected: PASS (all tests)

**Step 5: Commit**

```bash
git add tests/test_model_manager.py src/watermark_remover/model_manager.py
git commit -m "feat: add ModelManager for LaMa model download and management"
```

---

## Task 4: 建立 LamaInpainter（測試先行）

**Files:**
- Create: `tests/test_lama_inpainter.py`
- Create: `src/watermark_remover/lama_inpainter.py`

**Step 1: 建立 LamaInpainter 測試**

```python
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
```

**Step 2: 執行測試確認失敗**

Run: `pytest tests/test_lama_inpainter.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'watermark_remover.lama_inpainter'"

**Step 3: 實作 LamaInpainter**

```python
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
```

**Step 4: 執行測試確認通過**

Run: `pytest tests/test_lama_inpainter.py -v`
Expected: PASS (all tests)

**Step 5: Commit**

```bash
git add tests/test_lama_inpainter.py src/watermark_remover/lama_inpainter.py
git commit -m "feat: add LamaInpainter for AI-based image inpainting"
```

---

## Task 5: 修改 ImageProcessor 整合 LaMa

**Files:**
- Create: `tests/test_image_processor_lama.py`
- Modify: `src/watermark_remover/image_processor.py`

**Step 1: 建立 ImageProcessor LaMa 整合測試**

```python
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
```

**Step 2: 執行測試確認失敗**

Run: `pytest tests/test_image_processor_lama.py -v`
Expected: FAIL (InpaintMethod 沒有 LAMA)

**Step 3: 修改 ImageProcessor**

在 `src/watermark_remover/image_processor.py` 中進行以下修改：

```python
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
```

**Step 4: 執行測試確認通過**

Run: `pytest tests/test_image_processor_lama.py -v`
Expected: PASS (all tests)

**Step 5: 執行所有現有測試確保沒有破壞**

Run: `pytest -v`
Expected: PASS (all tests)

**Step 6: Commit**

```bash
git add src/watermark_remover/image_processor.py tests/test_image_processor_lama.py
git commit -m "feat: integrate LaMa into ImageProcessor with fallback to NS"
```

---

## Task 6: 修改 GUI - 新增算法選擇和模型下載

**Files:**
- Modify: `src/watermark_remover/gui_enhanced.py`

**Step 1: 新增模型下載執行緒類別**

在 `gui_enhanced.py` 的 `ProcessingThread` 類別後面新增：

```python
class ModelDownloadThread(QThread):
    """下載模型的背景執行緒"""

    finished = Signal()
    error = Signal(str)
    progress = Signal(int, int)  # downloaded, total

    def __init__(self, model_manager):
        super().__init__()
        self.model_manager = model_manager

    def run(self):
        """執行下載"""
        try:
            self.model_manager.download_model(
                "lama",
                progress_callback=lambda downloaded, total: self.progress.emit(downloaded, total),
            )
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))
```

**Step 2: 修改 imports**

在檔案開頭新增：

```python
from .model_manager import ModelManager
from .image_processor import InpaintMethod
```

**Step 3: 修改 WatermarkRemoverGUI.__init__**

```python
def __init__(self):
    super().__init__()
    self.processor = ImageProcessor()
    self.model_manager = ModelManager()
    self.current_image: np.ndarray | None = None
    self.detected_mask: np.ndarray | None = None
    self.lama_available = False
    self.current_method = InpaintMethod.LAMA

    # 歷史記錄
    self.history: deque = deque(maxlen=10)
    self.redo_stack: list = []

    self.setup_ui()
    self.setup_shortcuts()

    # 啟動時檢查模型
    self.check_model_on_startup()
```

**Step 4: 新增算法選擇 UI（修改 setup_ui）**

在 `zoom_layout` 之後新增算法選擇：

```python
# 算法選擇列
algorithm_layout = QHBoxLayout()
algorithm_layout.addWidget(QLabel("修復算法:"))

self.algorithm_combo = QComboBox()
self.algorithm_combo.addItems([
    "LaMa (AI) - 推薦，適合大面積",
    "NS - 快速，適合小面積",
    "Telea - 最快",
    "Hybrid - 混合",
])
self.algorithm_combo.setCurrentIndex(0)
self.algorithm_combo.currentIndexChanged.connect(self.on_algorithm_changed)
self.algorithm_combo.setMinimumWidth(250)
algorithm_layout.addWidget(self.algorithm_combo)

algorithm_layout.addStretch()

main_layout.addLayout(algorithm_layout)
```

需要在 imports 新增 `QComboBox`。

**Step 5: 新增算法相關方法**

```python
def on_algorithm_changed(self, index):
    """算法選擇改變"""
    methods = [InpaintMethod.LAMA, InpaintMethod.NS, InpaintMethod.TELEA, InpaintMethod.HYBRID]
    self.current_method = methods[index]
    self.processor.default_method = self.current_method
    self.status_bar.showMessage(f"已切換至 {self.algorithm_combo.currentText().split(' - ')[0]} 算法")

def check_model_on_startup(self):
    """啟動時檢查模型"""
    if self.model_manager.is_model_downloaded("lama"):
        self.lama_available = True
        self.status_bar.showMessage("LaMa 模型已就緒，請開啟圖像")
    else:
        reply = QMessageBox.question(
            self,
            "下載 AI 模型",
            "LaMa AI 模型尚未下載（約 200 MB）。\n"
            "此模型可大幅提升大面積浮水印的移除品質。\n\n"
            "是否立即下載？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.download_model_with_progress()
        else:
            self.lama_available = False
            self.update_algorithm_menu()
            self.status_bar.showMessage("請開啟圖像或拖放圖像檔案到視窗中")

def download_model_with_progress(self):
    """顯示下載進度"""
    self.download_progress = QProgressDialog(
        "正在下載 LaMa 模型...\n這可能需要幾分鐘",
        "取消",
        0,
        100,
        self,
    )
    self.download_progress.setWindowTitle("下載模型")
    self.download_progress.setWindowModality(Qt.WindowModality.WindowModal)
    self.download_progress.setMinimumDuration(0)
    self.download_progress.setValue(0)

    self.download_thread = ModelDownloadThread(self.model_manager)
    self.download_thread.progress.connect(self.on_download_progress)
    self.download_thread.finished.connect(self.on_download_finished)
    self.download_thread.error.connect(self.on_download_error)
    self.download_thread.start()

def on_download_progress(self, downloaded: int, total: int):
    """下載進度更新"""
    if total > 0:
        percent = int(downloaded / total * 100)
        self.download_progress.setValue(percent)
        mb_downloaded = downloaded / (1024 * 1024)
        mb_total = total / (1024 * 1024)
        self.download_progress.setLabelText(
            f"正在下載 LaMa 模型...\n{mb_downloaded:.1f} / {mb_total:.1f} MB"
        )

def on_download_finished(self):
    """下載完成"""
    self.download_progress.close()
    self.lama_available = True
    self.update_algorithm_menu()
    QMessageBox.information(self, "成功", "LaMa 模型下載完成！")
    self.status_bar.showMessage("LaMa 模型已就緒，請開啟圖像")

def on_download_error(self, error: str):
    """下載錯誤"""
    self.download_progress.close()
    QMessageBox.critical(self, "錯誤", f"下載模型失敗：{error}")
    self.lama_available = False
    self.update_algorithm_menu()

def update_algorithm_menu(self):
    """更新算法選單狀態"""
    if not self.lama_available:
        model = self.algorithm_combo.model()
        item = model.item(0)
        item.setEnabled(False)
        self.algorithm_combo.setItemText(0, "LaMa (AI) - 未下載")
        self.algorithm_combo.setCurrentIndex(1)
        self.current_method = InpaintMethod.NS
        self.processor.default_method = InpaintMethod.NS
    else:
        model = self.algorithm_combo.model()
        item = model.item(0)
        item.setEnabled(True)
        self.algorithm_combo.setItemText(0, "LaMa (AI) - 推薦，適合大面積")
```

**Step 6: 修改 ProcessingThread 支援算法選擇**

```python
class ProcessingThread(QThread):
    """處理圖像的背景執行緒"""

    finished = Signal(np.ndarray)
    error = Signal(str)
    progress = Signal(int)
    fallback = Signal(str)

    def __init__(
        self,
        processor: ImageProcessor,
        mask: np.ndarray | None,
        selection: tuple[int, int, int, int] | None,
        method: InpaintMethod = None,
    ):
        super().__init__()
        self.processor = processor
        self.mask = mask
        self.selection = selection
        self.method = method

    def run(self):
        """執行處理"""
        try:
            self.progress.emit(10)

            def progress_cb(value):
                self.progress.emit(value)

            def fallback_cb(msg):
                self.fallback.emit(msg)

            if self.mask is not None:
                result = self.processor.remove_watermark(
                    self.mask,
                    method=self.method,
                    progress_callback=progress_cb,
                    fallback_callback=fallback_cb,
                )
            elif self.selection:
                x, y, width, height = self.selection
                result = self.processor.remove_watermark_by_region(
                    x, y, width, height,
                    method=self.method,
                    progress_callback=progress_cb,
                    fallback_callback=fallback_cb,
                )
            else:
                self.error.emit("未選擇區域")
                return

            self.finished.emit(result)

        except Exception as e:
            self.error.emit(str(e))
```

**Step 7: 修改 remove_watermark 方法傳遞算法**

```python
def remove_watermark(self):
    """移除浮水印（使用進度對話框）"""
    if self.current_image is None:
        QMessageBox.warning(self, "警告", "請先載入圖像")
        return

    selection = self.image_label.get_selection_coords()

    if self.detected_mask is None and not selection:
        QMessageBox.warning(self, "警告", "請先選擇浮水印區域或使用自動偵測")
        return

    self.add_to_history()

    progress = QProgressDialog("正在移除浮水印...", "取消", 0, 100, self)
    progress.setWindowTitle("處理中")
    progress.setWindowModality(Qt.WindowModality.WindowModal)
    progress.setMinimumDuration(0)
    progress.setValue(0)

    self.processing_thread = ProcessingThread(
        self.processor,
        self.detected_mask,
        selection,
        self.current_method,  # 傳遞當前選擇的算法
    )

    self.processing_thread.progress.connect(progress.setValue)
    self.processing_thread.finished.connect(
        lambda result: self.on_processing_finished(result, progress)
    )
    self.processing_thread.error.connect(
        lambda error: self.on_processing_error(error, progress)
    )
    self.processing_thread.fallback.connect(
        lambda msg: self.status_bar.showMessage(msg)
    )

    self.processing_thread.start()
```

**Step 8: 執行測試**

Run: `python run_enhanced.py`
Expected: GUI 啟動，如果模型未下載會提示下載，算法選單可用

**Step 9: Commit**

```bash
git add src/watermark_remover/gui_enhanced.py
git commit -m "feat: add algorithm selector and model download UI to GUI"
```

---

## Task 7: 更新 __init__.py 匯出

**Files:**
- Modify: `src/watermark_remover/__init__.py`

**Step 1: 更新 __init__.py**

```python
"""圖像浮水印移除工具"""

from .image_processor import ImageProcessor, InpaintMethod
from .model_manager import ModelManager
from .lama_inpainter import LamaInpainter
from .exceptions import (
    LamaError,
    LamaModelNotFoundError,
    LamaDownloadError,
    LamaInferenceError,
    LamaMemoryError,
)

__all__ = [
    "ImageProcessor",
    "InpaintMethod",
    "ModelManager",
    "LamaInpainter",
    "LamaError",
    "LamaModelNotFoundError",
    "LamaDownloadError",
    "LamaInferenceError",
    "LamaMemoryError",
]
```

**Step 2: Commit**

```bash
git add src/watermark_remover/__init__.py
git commit -m "feat: export LaMa related modules from package"
```

---

## Task 8: 執行完整測試和驗證

**Files:** None (testing only)

**Step 1: 執行所有測試**

Run: `pytest -v`
Expected: PASS (all tests)

**Step 2: 手動測試 GUI**

Run: `python run_enhanced.py`
Expected:
1. 啟動時提示下載 LaMa 模型（如果未下載）
2. 下載進度正確顯示
3. 下載完成後 LaMa 選項可用
4. 可以切換不同算法
5. 使用 LaMa 處理大面積浮水印效果更好

**Step 3: 測試後備機制**

1. 刪除模型：`rm -rf ~/.cache/watermark-remover/models/lama`
2. 重新啟動 GUI，選擇「稍後」不下載
3. 驗證 LaMa 選項顯示為灰色且不可選
4. 驗證預設切換到 NS 算法

**Step 4: Commit final changes if any**

```bash
git status
# 如有任何修改
git add -A
git commit -m "fix: address any issues found during testing"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | 新增 PyTorch 依賴 | `pyproject.toml` |
| 2 | 建立例外類別 | `exceptions.py` |
| 3 | 建立 ModelManager | `model_manager.py`, `test_model_manager.py` |
| 4 | 建立 LamaInpainter | `lama_inpainter.py`, `test_lama_inpainter.py` |
| 5 | 整合 ImageProcessor | `image_processor.py`, `test_image_processor_lama.py` |
| 6 | 修改 GUI | `gui_enhanced.py` |
| 7 | 更新匯出 | `__init__.py` |
| 8 | 完整測試 | - |

Total: 8 tasks, ~30 steps
