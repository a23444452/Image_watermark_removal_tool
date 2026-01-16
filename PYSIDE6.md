# PySide6 GUI 介面

本專案使用 **PySide6**（Qt for Python）來建立圖形使用者介面，提供現代化和專業的使用體驗。

## 為什麼選擇 PySide6？

### 1. 現代化的外觀
- 使用 Qt 框架，提供原生的跨平台外觀
- 支援多種樣式主題（Fusion、Windows、macOS 等）
- 更好的高 DPI 螢幕支援

### 2. 更強大的功能
- 豐富的 Widget 元件
- 進階的繪圖能力（QPainter）
- 完整的事件處理系統
- 更好的滑鼠和鍵盤事件控制

### 3. 專業級應用
- Qt 被廣泛用於商業和專業應用
- 穩定且經過充分測試
- 活躍的社群和豐富的文件

### 4. 跨平台支援
- Windows、macOS、Linux 上外觀一致
- 自動適應系統主題
- 原生檔案對話框和訊息框

## 與 Tkinter 的比較

| 特性 | PySide6 | Tkinter |
|------|---------|---------|
| 外觀 | 現代化、專業 | 較簡單、老舊 |
| 效能 | 較好 | 普通 |
| 功能 | 豐富 | 基本 |
| 學習曲線 | 較陡峭 | 簡單 |
| 套件大小 | 較大（~100MB） | 很小（內建） |
| 跨平台 | 優秀 | 良好 |

## 主要元件說明

### ImageLabel (自定義 QLabel)
繼承自 `QLabel`，添加了：
- 滑鼠拖曳選擇功能
- 即時繪製選擇矩形
- 座標追蹤和管理

```python
class ImageLabel(QLabel):
    def mousePressEvent(self, event):
        # 處理滑鼠按下

    def mouseMoveEvent(self, event):
        # 處理滑鼠移動

    def paintEvent(self, event):
        # 自定義繪製
```

### WatermarkRemoverGUI (主視窗)
繼承自 `QMainWindow`，包含：
- 頂部按鈕工具列
- 捲動區域顯示大圖像
- 狀態列顯示訊息
- 檔案對話框和訊息框

## 主要改進

### 1. 更好的圖像顯示
- 使用 `QImage` 和 `QPixmap` 處理圖像
- 支援大圖像的捲動顯示
- 更流暢的即時繪製

### 2. 改進的使用者互動
- 即時的滑鼠拖曳回饋
- 更清晰的選擇矩形
- 原生的對話框體驗

### 3. 更好的事件處理
```python
# PySide6 的事件處理更加靈活
def mousePressEvent(self, event):
    if event.button() == Qt.MouseButton.LeftButton:
        # 處理左鍵點擊
        self.start_point = event.position().toPoint()
```

### 4. 自定義繪圖
```python
def paintEvent(self, event):
    super().paintEvent(event)
    painter = QPainter(self)
    pen = QPen(QColor(255, 0, 0), 2)
    painter.setPen(pen)
    painter.drawRect(self.selection_rect)
```

## 安裝 PySide6

```bash
# 使用 pip 安裝
pip install PySide6

# 或安裝整個專案
pip install -e .
```

## 執行應用程式

```bash
python run.py
```

## 樣式自定義

PySide6 支援多種樣式主題：

```python
# 在 run_gui() 中設定
app = QApplication(sys.argv)
app.setStyle("Fusion")  # 現代化的 Fusion 風格

# 其他可用樣式：
# - "Windows" - Windows 風格
# - "macOS" - macOS 風格
# - "WindowsVista" - Windows Vista/7 風格
```

## 進階自定義

### CSS 樣式表
可以使用 Qt Style Sheets (QSS) 自定義外觀：

```python
widget.setStyleSheet("""
    QPushButton {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px;
    }
    QPushButton:hover {
        background-color: #45a049;
    }
""")
```

### 自定義顏色
```python
info_label.setStyleSheet("color: blue; padding: 10px;")
scroll_area.setStyleSheet("background-color: #555555;")
```

## 效能考量

PySide6 在以下方面表現更好：
- 大圖像的顯示和捲動
- 即時繪圖和動畫
- 複雜的使用者互動
- 多視窗應用

## 學習資源

- [PySide6 官方文件](https://doc.qt.io/qtforpython/)
- [Qt for Python 教學](https://doc.qt.io/qtforpython/tutorials/index.html)
- [Qt Widget 範例](https://doc.qt.io/qtforpython/examples/index.html)

## 授權

PySide6 使用 LGPL 授權，可免費用於商業和非商業專案。
