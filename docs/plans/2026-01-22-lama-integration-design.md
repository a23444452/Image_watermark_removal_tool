# LaMa 模型整合設計文件

**建立日期**: 2026-01-22
**狀態**: 待實作
**目標**: 整合 LaMa AI 模型以提升大面積浮水印移除品質

---

## 1. 背景與目標

### 1.1 問題描述

目前專案使用 OpenCV 的 inpainting 算法（Telea、NS）處理浮水印移除。這些算法對小面積浮水印效果良好，但對大面積浮水印（> 100px）有以下限制：

- 修復區域出現模糊
- 紋理不連續
- 邊緣不自然

### 1.2 目標

- 整合 LaMa (Large Mask Inpainting) AI 模型
- 提升大面積浮水印的移除品質
- 維持 3-10 秒的處理時間
- 支援 Apple Silicon MPS 加速

### 1.3 適用場景

- AI 生成圖像（Midjourney、DALL-E、Stable Diffusion 等）
- 大面積浮水印（> 100px）
- 複雜背景上的浮水印

---

## 2. 技術方案

### 2.1 LaMa 模型簡介

**LaMa (Large Mask Inpainting)** 是由 Samsung AI Center 開發的圖像修復模型，專為大面積遮罩設計。

| 項目 | 規格 |
|------|------|
| 模型名稱 | big-lama |
| 模型大小 | ~200 MB |
| 輸入 | RGB 圖像 + 二值遮罩 |
| 輸出 | 修復後的 RGB 圖像 |
| 授權 | Apache 2.0 |

### 2.2 架構設計

```
現有架構                          新增架構
┌─────────────────────┐          ┌─────────────────────┐
│   ImageProcessor    │          │   ModelManager      │ ← 新增
│   ├── Telea         │          │   └── 模型下載/管理   │
│   ├── NS            │          └─────────────────────┘
│   ├── Hybrid        │                    │
│   ├── Advanced      │                    ▼
│   └── LaMa ─────────┼────────► ┌─────────────────────┐
└─────────────────────┘          │   LamaInpainter     │ ← 新增
                                 │   └── 模型推論       │
                                 └─────────────────────┘
```

### 2.3 模組職責

| 模組 | 職責 |
|------|------|
| `ModelManager` | 模型下載、版本管理、路徑管理 |
| `LamaInpainter` | 模型載入、推論、MPS 加速 |
| `ImageProcessor` | 統一調用介面、算法切換 |

---

## 3. 詳細設計

### 3.1 模型管理 (`model_manager.py`)

#### 3.1.1 模型儲存位置

```
~/.cache/watermark-remover/
└── models/
    └── lama/
        └── big-lama.pt  (~200 MB)
```

#### 3.1.2 類別設計

```python
class ModelManager:
    """管理 AI 模型的下載與儲存"""

    MODEL_URL = "https://huggingface.co/..."  # 待確認確切 URL
    MODEL_DIR = Path.home() / ".cache" / "watermark-remover" / "models"

    def __init__(self):
        self.model_dir = self.MODEL_DIR
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def is_model_downloaded(self, model_name: str = "lama") -> bool:
        """檢查模型是否已下載"""
        pass

    def download_model(
        self,
        model_name: str = "lama",
        progress_callback: Callable[[int, int], None] = None
    ) -> Path:
        """下載模型，支援進度回調"""
        pass

    def get_model_path(self, model_name: str = "lama") -> Path:
        """取得模型路徑"""
        pass

    def get_model_size(self, model_name: str = "lama") -> int:
        """取得模型大小（bytes）"""
        pass
```

### 3.2 LaMa 推論器 (`lama_inpainter.py`)

#### 3.2.1 類別設計

```python
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
        self.device = self._detect_device(device)
        self._load_model(model_path)

    def _detect_device(self, device: str) -> torch.device:
        """偵測最佳運算設備"""
        if device == "auto":
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)

    def _load_model(self, model_path: Path) -> None:
        """載入模型"""
        pass

    def inpaint(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """
        執行圖像修復

        Args:
            image: BGR 圖像 (H, W, 3)
            mask: 二值遮罩 (H, W)，255 表示要修復的區域

        Returns:
            修復後的 BGR 圖像
        """
        pass

    def unload_model(self) -> None:
        """釋放模型記憶體"""
        pass
```

#### 3.2.2 推論流程

```
輸入圖像 (BGR)
    ↓
轉換為 RGB
    ↓
縮放至模型輸入尺寸 (如需要)
    ↓
正規化 [0, 1]
    ↓
轉換為 Tensor
    ↓
模型推論 (MPS/CPU)
    ↓
反正規化 [0, 255]
    ↓
縮放回原始尺寸 (如需要)
    ↓
轉換為 BGR
    ↓
輸出圖像
```

### 3.3 ImageProcessor 修改

#### 3.3.1 新增列舉值

```python
class InpaintMethod(Enum):
    TELEA = "telea"
    NS = "ns"
    HYBRID = "hybrid"
    ADVANCED = "advanced"
    LAMA = "lama"  # 新增
```

#### 3.3.2 修改預設值

```python
# 舊
DEFAULT_METHOD = InpaintMethod.NS

# 新
DEFAULT_METHOD = InpaintMethod.LAMA
```

#### 3.3.3 新增方法

```python
def remove_watermark_lama(
    self,
    mask: np.ndarray,
    progress_callback: Callable[[int], None] = None
) -> np.ndarray:
    """
    使用 LaMa 模型移除浮水印

    Args:
        mask: 浮水印遮罩
        progress_callback: 進度回調 (0-100)

    Returns:
        處理後的圖像
    """
    pass
```

#### 3.3.4 後備機制

```python
def remove_watermark(self, mask: np.ndarray, method: InpaintMethod = None) -> np.ndarray:
    method = method or self.default_method

    if method == InpaintMethod.LAMA:
        try:
            return self.remove_watermark_lama(mask)
        except LamaUnavailableError:
            # 自動降級至 NS
            self._show_fallback_warning()
            return self.remove_watermark_ns(mask)

    # ... 其他方法
```

### 3.4 GUI 修改 (`gui_enhanced.py`)

#### 3.4.1 啟動檢查流程

```python
def check_model_on_startup(self):
    """啟動時檢查模型"""
    if not self.model_manager.is_model_downloaded("lama"):
        reply = QMessageBox.question(
            self,
            "下載 AI 模型",
            "LaMa AI 模型尚未下載（約 200 MB）。\n"
            "此模型可大幅提升大面積浮水印的移除品質。\n\n"
            "是否立即下載？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )

        if reply == QMessageBox.Yes:
            self.download_model_with_progress()
        else:
            self.lama_available = False
            self.update_algorithm_menu()
```

#### 3.4.2 下載進度對話框

```python
def download_model_with_progress(self):
    """顯示下載進度"""
    progress = QProgressDialog(
        "正在下載 LaMa 模型...",
        "取消",
        0, 100,
        self
    )
    progress.setWindowModality(Qt.WindowModal)

    # 使用 QThread 背景下載
    self.download_thread = ModelDownloadThread(self.model_manager)
    self.download_thread.progress.connect(progress.setValue)
    self.download_thread.finished.connect(self.on_download_finished)
    self.download_thread.error.connect(self.on_download_error)
    self.download_thread.start()
```

#### 3.4.3 算法選擇下拉選單

```python
def create_algorithm_selector(self):
    """建立算法選擇下拉選單"""
    self.algorithm_combo = QComboBox()
    self.algorithm_combo.addItems([
        "LaMa (AI) - 推薦，適合大面積",
        "NS - 快速，適合小面積",
        "Telea - 最快",
        "Hybrid - 混合",
        "Advanced - 多層次處理"
    ])
    self.algorithm_combo.setCurrentIndex(0)  # 預設 LaMa
    self.algorithm_combo.currentIndexChanged.connect(self.on_algorithm_changed)
```

#### 3.4.4 LaMa 不可用時的 UI 狀態

```python
def update_algorithm_menu(self):
    """更新算法選單狀態"""
    if not self.lama_available:
        # LaMa 選項顯示為灰色
        model = self.algorithm_combo.model()
        item = model.item(0)
        item.setEnabled(False)
        item.setText("LaMa (AI) - 未下載")

        # 切換預設為 NS
        self.algorithm_combo.setCurrentIndex(1)
```

---

## 4. 檔案變更清單

| 檔案 | 變更類型 | 說明 |
|------|----------|------|
| `src/watermark_remover/model_manager.py` | 新增 | 模型下載與管理 |
| `src/watermark_remover/lama_inpainter.py` | 新增 | LaMa 模型載入與推論 |
| `src/watermark_remover/image_processor.py` | 修改 | 新增 LAMA 方法、改預設值 |
| `src/watermark_remover/gui_enhanced.py` | 修改 | 新增算法選單、啟動檢查 |
| `pyproject.toml` | 修改 | 新增 PyTorch 依賴 |
| `tests/test_lama.py` | 新增 | LaMa 相關測試 |

---

## 5. 依賴變更

### 5.1 新增依賴

```toml
# pyproject.toml
dependencies = [
    # 現有依賴...
    "torch>=2.0.0",
    "torchvision>=0.15.0",
]
```

### 5.2 Apple Silicon 安裝說明

```bash
# PyTorch 會自動偵測 Apple Silicon 並使用 MPS
pip install torch torchvision
```

---

## 6. 錯誤處理

### 6.1 自定義例外

```python
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

### 6.2 錯誤處理策略

| 錯誤 | 處理方式 |
|------|----------|
| `LamaModelNotFoundError` | 提示下載模型 |
| `LamaDownloadError` | 顯示錯誤訊息，提供重試按鈕 |
| `LamaInferenceError` | 記錄日誌，降級至 NS 算法 |
| `LamaMemoryError` | 提示記憶體不足，降級至 NS 算法 |

---

## 7. 測試計畫

### 7.1 單元測試

- [ ] `ModelManager.is_model_downloaded()` 正確偵測
- [ ] `ModelManager.download_model()` 下載成功
- [ ] `LamaInpainter._detect_device()` 正確偵測 MPS
- [ ] `LamaInpainter.inpaint()` 輸出尺寸正確

### 7.2 整合測試

- [ ] 完整流程：選擇區域 → LaMa 處理 → 儲存
- [ ] 後備機制：LaMa 失敗 → 自動切換 NS
- [ ] 啟動檢查：未下載 → 提示 → 下載 → 可用

### 7.3 效能測試

- [ ] 處理時間 < 10 秒（1024x1024 圖像）
- [ ] 記憶體使用 < 2 GB
- [ ] MPS 加速正常運作

---

## 8. 實作優先順序

1. **Phase 1**: 模型管理（下載、儲存）
2. **Phase 2**: LaMa 推論器
3. **Phase 3**: ImageProcessor 整合
4. **Phase 4**: GUI 整合（啟動檢查、算法選單）
5. **Phase 5**: 測試與調優

---

## 9. 風險與緩解

| 風險 | 可能性 | 緩解措施 |
|------|--------|----------|
| MPS 相容性問題 | 中 | 提供 CPU 後備方案 |
| 模型下載失敗 | 低 | 多鏡像源、斷點續傳 |
| 記憶體不足 | 中 | 大圖分塊處理、及時釋放 |
| 處理時間過長 | 低 | 圖像縮放、進度顯示 |

---

## 10. 成功指標

- [ ] 大面積浮水印（200x200px）移除後無明顯痕跡
- [ ] 處理時間 3-10 秒（1024x1024 圖像，Apple Silicon）
- [ ] 首次啟動模型下載流程順暢
- [ ] 後備機制正常運作
