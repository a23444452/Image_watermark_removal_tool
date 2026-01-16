#!/usr/bin/env python3
"""功能性測試 - 測試浮水印移除功能"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import cv2
import numpy as np
from watermark_remover.image_processor import ImageProcessor


def test_watermark_removal():
    """測試完整的浮水印移除流程"""

    print("\n" + "=" * 60)
    print("測試浮水印移除功能")
    print("=" * 60 + "\n")

    # 檢查測試圖像是否存在
    test_image_path = Path(__file__).parent / "test_image.png"

    if not test_image_path.exists():
        print("測試圖像不存在，正在建立...")
        import subprocess

        subprocess.run([sys.executable, "create_test_image.py"])

    # 初始化處理器
    processor = ImageProcessor()
    print("1. 初始化圖像處理器... ✓")

    # 載入測試圖像
    if processor.load_image(test_image_path):
        print(f"2. 載入測試圖像... ✓")
        print(f"   圖像尺寸: {processor.image.shape}")
    else:
        print("✗ 無法載入測試圖像")
        return False

    # 測試自動偵測浮水印
    print("\n3. 測試自動偵測浮水印...")
    mask = processor.auto_detect_watermark()

    if mask is not None:
        white_pixels = cv2.countNonZero(mask)
        print(f"   ✓ 偵測到浮水印區域")
        print(f"   偵測到的像素數: {white_pixels}")

        # 使用偵測到的遮罩移除浮水印
        print("\n4. 使用自動偵測的遮罩移除浮水印...")
        result_auto = processor.remove_watermark(mask)

        # 儲存結果
        output_auto = Path(__file__).parent / "test_output_auto.png"
        if processor.save_image(result_auto, output_auto):
            print(f"   ✓ 已儲存結果: {output_auto}")
        else:
            print("   ✗ 儲存失敗")
    else:
        print("   ⚠ 未偵測到浮水印（這可能是正常的）")

    # 重新載入原始圖像以進行手動測試
    processor.load_image(test_image_path)

    # 測試手動選擇區域移除
    print("\n5. 測試手動選擇區域移除浮水印...")
    # 選擇右下角區域（假設浮水印在這裡）
    height, width = processor.image.shape[:2]
    x = width - 250
    y = height - 80
    w = 230
    h = 60

    print(f"   選擇區域: x={x}, y={y}, width={w}, height={h}")

    result_manual = processor.remove_watermark_by_region(x, y, w, h)
    print("   ✓ 浮水印移除完成")

    # 儲存手動移除的結果
    output_manual = Path(__file__).parent / "test_output_manual.png"
    if processor.save_image(result_manual, output_manual):
        print(f"   ✓ 已儲存結果: {output_manual}")
    else:
        print("   ✗ 儲存失敗")

    # 比較原始圖像和處理後的圖像
    print("\n6. 比較結果...")
    original = processor.original
    difference = cv2.absdiff(original, result_manual)
    diff_sum = np.sum(difference)

    if diff_sum > 0:
        print(f"   ✓ 圖像已被修改（差異總和: {diff_sum}）")
    else:
        print("   ⚠ 圖像未改變")

    print("\n" + "=" * 60)
    print("✓ 功能測試完成！")
    print("=" * 60)
    print("\n產生的檔案：")
    print(f"  - 原始圖像: {test_image_path}")
    if mask is not None:
        print(f"  - 自動偵測結果: {output_auto}")
    print(f"  - 手動移除結果: {output_manual}")

    print("\n你可以比較這些圖像來檢視浮水印移除效果。")

    return True


def main():
    """執行測試"""
    try:
        success = test_watermark_removal()
        return 0 if success else 1
    except Exception as e:
        print(f"\n✗ 測試過程中發生錯誤: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
