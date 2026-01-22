# 圖像浮水印移除工具

一個現代化的 GUI 工具，用於移除圖像中的浮水印（如 Gemini、ChatGPT 生成的圖像浮水印）。

> 🚀 **重大更新：LaMa AI 模型整合！** 現在支援 LaMa (Large Mask Inpainting) AI 模型，大幅提升大面積浮水印的移除品質。
>
> 🎉 **增強版 GUI** 提供快捷鍵、拖放檔案、圖像縮放、撤銷/重做等專業功能。
>
> ⚡ **多種修復算法** 包含 LaMa (AI)、NS、Telea、混合算法，滿足不同需求。

## 安裝

### 方法 1：使用 pip 安裝專案（推薦）

```bash
cd watermark-remover
pip install -e .
```

### 方法 2：手動安裝相依套件

```bash
pip install opencv-python numpy Pillow PySide6 torch torchvision
```

## 使用方式

### 增強版（推薦）✨

```bash
python run_enhanced.py
```

首次啟動時會提示下載 LaMa 模型（約 200 MB），下載後即可使用 AI 驅動的浮水印移除功能。

### 基礎版

```bash
python run.py
```

## 功能特色

### 🤖 LaMa AI 模型（新增）

- **大面積浮水印移除** - LaMa 專為大面積遮罩設計，效果遠優於傳統算法
- **自動下載** - 首次使用時自動下載模型（約 200 MB）
- **Apple Silicon 加速** - 支援 MPS 加速，在 M1/M2/M3/M4 Mac 上快速運行
- **智能後備** - LaMa 不可用時自動切換至 NS 算法

### 🎨 修復算法

| 算法 | 說明 | 適用場景 |
|------|------|----------|
| **LaMa (AI)** | 大型遮罩修復 AI 模型 | 大面積浮水印（推薦）⭐ |
| **NS** | Navier-Stokes 流體力學法 | 小面積浮水印 |
| **Telea** | 快速行進法 | 需要快速處理 |
| **Hybrid** | 混合算法 | 平衡品質與速度 |

### 🖥️ 增強版 GUI

- ⌨️ **快捷鍵支援** - Ctrl+O 開啟、Ctrl+S 儲存、Ctrl+R 移除、Ctrl+Z 撤銷
- 📂 **拖放檔案** - 直接拖放圖像到視窗開啟
- 🔍 **圖像縮放** - Ctrl+滾輪縮放，支援 20%-500%
- ↶↷ **撤銷/重做** - 最多保存 10 個歷史狀態
- ⏳ **處理進度** - 即時顯示處理進度
- 🎛️ **算法選擇** - 下拉選單切換修復算法

### 🔧 基本功能

- 🖼️ 現代化 Qt 介面（PySide6）
- 🎯 **手動框選浮水印區域** - 滑鼠拖曳選擇要移除的區域
- 🤖 **自動偵測浮水印** - 智能偵測圖像角落的浮水印
- 🔄 **智能優先順序** - 手動框選優先於自動偵測結果
- 💾 儲存處理後的圖像
- ↩️ 重置功能

## 技術棧

- **PyTorch** - LaMa AI 模型推論
- **OpenCV** - 傳統圖像處理和修復算法
- **PySide6 (Qt)** - 現代化 GUI 介面
- **NumPy** - 數值運算
- **Pillow** - 圖像格式支援

## 系統需求

- Python 3.12+
- macOS（Apple Silicon 支援 MPS 加速）/ Windows / Linux
- 約 500 MB 硬碟空間（含 LaMa 模型）

## 模型管理

LaMa 模型儲存在 `~/.cache/watermark-remover/models/lama/`。

### 手動刪除模型

```bash
rm -rf ~/.cache/watermark-remover/models/lama
```

重新啟動 GUI 後會提示重新下載。

## 測試

```bash
# 執行所有測試
pytest -v

# 測試不同算法效果
python test_algorithm_comparison.py
```

## 文件

- [增強功能說明](ENHANCED_FEATURES.md)
- [演算法改進說明](ALGORITHM_IMPROVEMENTS.md)
- [設計文件](docs/plans/2026-01-22-lama-integration-design.md)

## 授權

MIT License
