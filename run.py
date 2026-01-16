#!/usr/bin/env python3
"""執行腳本"""

import sys
from pathlib import Path

# 添加 src 到路徑
sys.path.insert(0, str(Path(__file__).parent / "src"))

from watermark_remover.main import main

if __name__ == "__main__":
    main()
