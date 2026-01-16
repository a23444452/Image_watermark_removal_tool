#!/usr/bin/env python3
"""測試和比較不同修復算法的效果"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import cv2
import numpy as np
from watermark_remover.image_processor import ImageProcessor, InpaintMethod


def test_algorithms():
    """測試不同算法的效果"""

    print("\n" + "=" * 60)
    print("測試修復算法比較")
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

    # 建立測試遮罩（右下角浮水印區域）
    height, width = processor.image.shape[:2]
    x = width - 250
    y = height - 80
    w = 230
    h = 60

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y : y + h, x : x + w] = 255

    print(f"\n3. 建立測試遮罩: x={x}, y={y}, width={w}, height={h}")

    # 測試不同算法
    print("\n4. 測試不同修復算法...")
    print("-" * 60)

    # Telea 算法
    print("\n[1] Telea 算法（快速行進法）")
    print("    特點：速度快，適合小面積浮水印")
    result_telea = processor.remove_watermark(mask, InpaintMethod.TELEA, radius=5)
    output_telea = Path(__file__).parent / "test_output_telea.png"
    processor.save_image(result_telea, output_telea)
    print(f"    ✓ 已儲存：{output_telea}")

    # NS 算法
    print("\n[2] NS 算法（Navier-Stokes）")
    print("    特點：品質較好，處理較慢，適合較大面積")
    result_ns = processor.remove_watermark(mask, InpaintMethod.NS, radius=5)
    output_ns = Path(__file__).parent / "test_output_ns.png"
    processor.save_image(result_ns, output_ns)
    print(f"    ✓ 已儲存：{output_ns}")

    # 混合算法
    print("\n[3] 混合算法（NS + Telea）")
    print("    特點：結合兩種算法優勢")
    result_hybrid = processor.remove_watermark(mask, InpaintMethod.HYBRID, radius=5)
    output_hybrid = Path(__file__).parent / "test_output_hybrid.png"
    processor.save_image(result_hybrid, output_hybrid)
    print(f"    ✓ 已儲存：{output_hybrid}")

    # 進階算法
    print("\n[4] 進階處理（多層次處理）")
    print("    特點：使用多種技術組合，品質最好但最慢")
    result_advanced = processor.remove_watermark_advanced(mask, radius=7)
    output_advanced = Path(__file__).parent / "test_output_advanced.png"
    processor.save_image(result_advanced, output_advanced)
    print(f"    ✓ 已儲存：{output_advanced}")

    # 測試不同半徑
    print("\n5. 測試不同修復半徑（使用 NS 算法）...")
    print("-" * 60)

    for radius in [3, 5, 7, 10]:
        print(f"\n半徑 = {radius}")
        result = processor.remove_watermark(mask, InpaintMethod.NS, radius=radius)
        output = (
            Path(__file__).parent / f"test_output_ns_radius{radius}.png"
        )
        processor.save_image(result, output)
        print(f"✓ 已儲存：{output}")

    # 比較所有方法
    print("\n6. 生成比較結果...")
    results = processor.compare_methods(mask, radius=5)
    for name, result in results.items():
        output = (
            Path(__file__).parent
            / f"compare_{name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
        )
        processor.save_image(result, output)
        print(f"✓ {name}: {output}")

    print("\n" + "=" * 60)
    print("✓ 算法比較測試完成！")
    print("=" * 60)

    print("\n生成的檔案：")
    print("基本算法：")
    print(f"  - Telea: {output_telea}")
    print(f"  - NS (推薦): {output_ns}")
    print(f"  - 混合: {output_hybrid}")
    print(f"  - 進階: {output_advanced}")

    print("\n不同半徑（NS 算法）：")
    for radius in [3, 5, 7, 10]:
        print(f"  - 半徑 {radius}: test_output_ns_radius{radius}.png")

    print("\n建議：")
    print("1. 比較這些圖像的視覺效果")
    print("2. NS 算法（半徑 5-7）通常提供最佳品質")
    print("3. 進階處理品質最好但速度較慢")
    print("4. Telea 速度最快但品質稍差")

    return True


def calculate_quality_metrics():
    """計算品質指標（可選）"""
    print("\n" + "=" * 60)
    print("品質分析（與原始圖像比較）")
    print("=" * 60)

    # 這裡可以添加 PSNR、SSIM 等品質指標的計算
    # 但需要知道原始無浮水印的圖像作為參考
    print("\n提示：要計算準確的品質指標，需要原始無浮水印圖像作為參考")


def main():
    """執行測試"""
    try:
        success = test_algorithms()
        if success:
            # calculate_quality_metrics()  # 可選
            pass
        return 0 if success else 1
    except Exception as e:
        print(f"\n✗ 測試過程中發生錯誤: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
