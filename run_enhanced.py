#!/usr/bin/env python3
"""啟動增強版 GUI 的腳本"""

import sys
from pathlib import Path

# 添加 src 到路徑
sys.path.insert(0, str(Path(__file__).parent / "src"))

from watermark_remover.gui_enhanced import run_gui

if __name__ == "__main__":
    run_gui()
