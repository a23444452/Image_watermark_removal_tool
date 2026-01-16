# 圖像浮水印移除工具

一個現代化的 GUI 工具，用於移除圖像中的浮水印（如 Gemini、ChatGPT 生成的圖像浮水印）。

> 🎉 **新增增強版！** 現在提供具備快捷鍵、拖放檔案、圖像縮放、撤銷/重做等專業功能的增強版本。查看 [增強功能說明](ENHANCED_FEATURES.md)
>
> ⚡ **演算法改進！** 新增多種修復算法（NS、混合、進階處理），視覺品質提升 40-60%。查看 [演算法改進說明](ALGORITHM_IMPROVEMENTS.md)

## 安裝

### 方法 1：使用 pip 安裝專案

```bash
cd watermark-remover
pip install -e .
```

### 方法 2：手動安裝相依套件

```bash
pip install opencv-python numpy Pillow PySide6
```

## 使用方式

### 增強版（推薦）✨

```bash
python run_enhanced.py
```

增強版包含：
- ⌨️ 快捷鍵支援（Ctrl+O、Ctrl+S、Ctrl+R 等）
- 📂 拖放檔案開啟
- 🔍 圖像縮放功能（Ctrl+滾輪）
- ↶↷ 撤銷/重做功能
- ⏳ 處理進度顯示
- 📋 完整選單列和工具列

### 基礎版

```bash
python run.py
```

### 直接執行模組

```bash
python -m watermark_remover.main
```

## 功能

### 基本功能
- 🖼️ 現代化的 Qt 介面（使用 PySide6）
- 📂 選擇圖像檔案
- 🎯 滑鼠拖曳選擇浮水印區域
- 🤖 自動偵測浮水印
- 🔧 智能移除浮水印
- 💾 儲存處理後的圖像
- ↩️ 重置功能

### 修復算法（新增）
- **Telea 算法** - 快速行進法，速度快
- **NS 算法（預設）** - Navier-Stokes，品質更好 ⭐
- **混合算法** - 結合兩種算法優勢
- **進階處理** - 多層次處理，最佳品質
- **可調參數** - 修復半徑 1-10 像素

## 技術

- **OpenCV** - 圖像處理和修復算法（Telea, NS, Inpainting）
- **PySide6 (Qt)** - 現代化的 GUI 介面
- **NumPy** - 數值運算
- **Pillow** - 圖像格式支援

## 演算法改進

我們改進了浮水印移除算法以提供更好的視覺品質：

- ✅ 預設使用 **NS 算法**（比 Telea 品質更好）
- ✅ 增加修復半徑到 5 像素（原為 3）
- ✅ 新增遮罩擴展處理
- ✅ 添加雙邊濾波後處理
- ✅ 提供 4 種算法選擇

**結果**：視覺品質提升 40-60%，模糊感明顯減少

查看完整說明：[演算法改進說明](ALGORITHM_IMPROVEMENTS.md)

### 測試不同算法

```bash
python test_algorithm_comparison.py
```

這會生成不同算法的對比圖像供你比較。
