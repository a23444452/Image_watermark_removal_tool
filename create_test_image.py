#!/usr/bin/env python3
"""建立測試圖像並加上模擬浮水印"""

import cv2
import numpy as np
from pathlib import Path


def create_test_image():
    """建立一個帶有浮水印的測試圖像"""

    # 建立一個漸層背景圖像
    height, width = 600, 800
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # 建立漸層背景
    for i in range(height):
        for j in range(width):
            image[i, j] = [
                int(100 + (i / height) * 100),  # B
                int(150 + (j / width) * 50),  # G
                int(200 - (i / height) * 50),  # R
            ]

    # 添加一些圖案
    # 繪製圓形
    cv2.circle(image, (400, 300), 100, (255, 200, 100), -1)
    cv2.circle(image, (400, 300), 100, (0, 0, 0), 2)

    # 繪製矩形
    cv2.rectangle(image, (200, 200), (300, 350), (100, 255, 100), -1)
    cv2.rectangle(image, (200, 200), (300, 350), (0, 0, 0), 2)

    # 添加浮水印（右下角）
    font = cv2.FONT_HERSHEY_SIMPLEX
    watermark_text = "AI Generated"
    font_scale = 1.2
    thickness = 2

    # 取得文字大小
    (text_width, text_height), baseline = cv2.getTextSize(
        watermark_text, font, font_scale, thickness
    )

    # 浮水印位置（右下角）
    x = width - text_width - 20
    y = height - 20

    # 添加半透明背景
    overlay = image.copy()
    cv2.rectangle(
        overlay,
        (x - 10, y - text_height - 10),
        (x + text_width + 10, y + 10),
        (255, 255, 255),
        -1,
    )
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

    # 添加浮水印文字
    cv2.putText(image, watermark_text, (x, y), font, font_scale, (0, 0, 0), thickness)

    # 儲存圖像
    output_path = Path(__file__).parent / "test_image.png"
    cv2.imwrite(str(output_path), image)
    print(f"✓ 測試圖像已建立：{output_path}")
    print(f"  尺寸：{width}x{height}")
    print(f"  浮水印位置：右下角")
    print("\n使用說明：")
    print("1. 執行：python run.py")
    print("2. 開啟圖像：test_image.png")
    print("3. 選擇右下角的浮水印區域")
    print("4. 點擊「移除浮水印」")
    print("5. 儲存處理後的圖像")

    return output_path


if __name__ == "__main__":
    create_test_image()
