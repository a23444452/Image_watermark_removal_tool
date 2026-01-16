#!/usr/bin/env python3
"""基本功能測試腳本"""

import sys
from pathlib import Path

# 添加 src 到路徑
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
from watermark_remover.image_processor import ImageProcessor


def test_image_processor():
    """測試圖像處理器基本功能"""
    print("=" * 60)
    print("測試圖像處理器")
    print("=" * 60)

    processor = ImageProcessor()
    print("✓ ImageProcessor 初始化成功")

    # 建立測試圖像（模擬一個簡單的圖像）
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    test_image[:, :] = [100, 150, 200]  # 填充顏色
    print("✓ 測試圖像建立成功")

    # 測試圖像形狀
    assert test_image.shape == (100, 100, 3), "圖像形狀錯誤"
    print("✓ 圖像形狀驗證通過")

    # 設定圖像
    processor.image = test_image
    processor.original = test_image.copy()
    print("✓ 圖像載入成功")

    # 測試移除浮水印功能（使用小區域）
    try:
        result = processor.remove_watermark_by_region(10, 10, 20, 20)
        assert result.shape == test_image.shape, "處理後圖像形狀不符"
        print("✓ 浮水印移除功能測試通過")
    except Exception as e:
        print(f"✗ 浮水印移除測試失敗: {e}")
        return False

    print("\n所有圖像處理器測試通過！\n")
    return True


def test_gui_imports():
    """測試 GUI 模組匯入"""
    print("=" * 60)
    print("測試 GUI 模組")
    print("=" * 60)

    try:
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QImage, QPixmap

        print("✓ PySide6.QtWidgets 匯入成功")
        print("✓ PySide6.QtCore 匯入成功")
        print("✓ PySide6.QtGui 匯入成功")

        from watermark_remover.gui import ImageLabel, WatermarkRemoverGUI

        print("✓ ImageLabel 類別匯入成功")
        print("✓ WatermarkRemoverGUI 類別匯入成功")

        print("\n所有 GUI 模組匯入測試通過！\n")
        return True

    except Exception as e:
        print(f"✗ GUI 模組測試失敗: {e}")
        return False


def test_main_module():
    """測試主模組"""
    print("=" * 60)
    print("測試主模組")
    print("=" * 60)

    try:
        from watermark_remover.main import main

        print("✓ main 函數匯入成功")
        print("\n主模組測試通過！\n")
        return True

    except Exception as e:
        print(f"✗ 主模組測試失敗: {e}")
        return False


def main():
    """執行所有測試"""
    print("\n" + "=" * 60)
    print("開始測試 PySide6 浮水印移除工具")
    print("=" * 60 + "\n")

    results = []

    # 測試圖像處理器
    results.append(("圖像處理器", test_image_processor()))

    # 測試 GUI 匯入
    results.append(("GUI 模組", test_gui_imports()))

    # 測試主模組
    results.append(("主模組", test_main_module()))

    # 顯示測試結果摘要
    print("=" * 60)
    print("測試結果摘要")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✓ 通過" if passed else "✗ 失敗"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n🎉 所有測試通過！程式可以正常運行。")
        print("\n執行程式：python run.py")
        return 0
    else:
        print("\n❌ 部分測試失敗，請檢查錯誤訊息。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
