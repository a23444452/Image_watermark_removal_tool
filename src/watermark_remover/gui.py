"""GUI 介面模組 - 使用 PySide6"""

from pathlib import Path
import sys

import cv2
import numpy as np
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QScrollArea,
    QFileDialog,
    QMessageBox,
    QStatusBar,
)
from PySide6.QtCore import Qt, QRect, QPoint
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor

from .image_processor import ImageProcessor


class ImageLabel(QLabel):
    """可繪製選擇區域的圖像標籤"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScaledContents(False)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.selecting = False
        self.start_point = QPoint()
        self.end_point = QPoint()
        self.selection_rect: QRect | None = None

    def mousePressEvent(self, event):
        """滑鼠按下事件"""
        if event.button() == Qt.MouseButton.LeftButton and self.pixmap():
            self.selecting = True
            self.start_point = event.position().toPoint()
            self.end_point = self.start_point
            self.update()

    def mouseMoveEvent(self, event):
        """滑鼠移動事件"""
        if self.selecting:
            self.end_point = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event):
        """滑鼠放開事件"""
        if event.button() == Qt.MouseButton.LeftButton and self.selecting:
            self.selecting = False
            self.end_point = event.position().toPoint()

            # 計算選擇區域
            x1 = min(self.start_point.x(), self.end_point.x())
            y1 = min(self.start_point.y(), self.end_point.y())
            x2 = max(self.start_point.x(), self.end_point.x())
            y2 = max(self.start_point.y(), self.end_point.y())

            width = x2 - x1
            height = y2 - y1

            if width > 5 and height > 5:  # 最小選擇區域
                self.selection_rect = QRect(x1, y1, width, height)
            else:
                self.selection_rect = None

            self.update()

    def paintEvent(self, event):
        """繪製事件"""
        super().paintEvent(event)

        if not self.pixmap():
            return

        painter = QPainter(self)
        pen = QPen(QColor(255, 0, 0), 2)
        painter.setPen(pen)

        # 繪製當前拖曳的矩形
        if self.selecting:
            rect = QRect(self.start_point, self.end_point).normalized()
            painter.drawRect(rect)

        # 繪製已確認的選擇區域
        elif self.selection_rect:
            painter.drawRect(self.selection_rect)

        painter.end()

    def clear_selection(self):
        """清除選擇區域"""
        self.selection_rect = None
        self.selecting = False
        self.update()

    def get_selection_coords(self) -> tuple[int, int, int, int] | None:
        """取得選擇區域座標 (x, y, width, height)"""
        if self.selection_rect:
            return (
                self.selection_rect.x(),
                self.selection_rect.y(),
                self.selection_rect.width(),
                self.selection_rect.height(),
            )
        return None


class WatermarkRemoverGUI(QMainWindow):
    """浮水印移除工具的主視窗"""

    def __init__(self):
        super().__init__()
        self.processor = ImageProcessor()
        self.current_image: np.ndarray | None = None
        self.detected_mask: np.ndarray | None = None

        self.setup_ui()

    def setup_ui(self):
        """設定 UI 元件"""
        self.setWindowTitle("圖像浮水印移除工具")
        self.setGeometry(100, 100, 1200, 800)

        # 中央 widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主佈局
        main_layout = QVBoxLayout(central_widget)

        # 頂部按鈕區
        button_layout = QHBoxLayout()

        self.btn_open = QPushButton("開啟圖像")
        self.btn_open.setMinimumHeight(40)
        self.btn_open.clicked.connect(self.load_image)
        button_layout.addWidget(self.btn_open)

        self.btn_auto_detect = QPushButton("自動偵測浮水印")
        self.btn_auto_detect.setMinimumHeight(40)
        self.btn_auto_detect.clicked.connect(self.auto_detect)
        button_layout.addWidget(self.btn_auto_detect)

        self.btn_remove = QPushButton("移除浮水印")
        self.btn_remove.setMinimumHeight(40)
        self.btn_remove.clicked.connect(self.remove_watermark)
        button_layout.addWidget(self.btn_remove)

        self.btn_save = QPushButton("儲存圖像")
        self.btn_save.setMinimumHeight(40)
        self.btn_save.clicked.connect(self.save_image)
        button_layout.addWidget(self.btn_save)

        self.btn_reset = QPushButton("重置")
        self.btn_reset.setMinimumHeight(40)
        self.btn_reset.clicked.connect(self.reset_image)
        button_layout.addWidget(self.btn_reset)

        main_layout.addLayout(button_layout)

        # 說明文字
        info_label = QLabel(
            "使用方式：1. 開啟圖像 2. 在圖像上拖曳選擇浮水印區域（或使用自動偵測）3. 點擊移除浮水印 4. 儲存結果"
        )
        info_label.setStyleSheet("color: blue; padding: 10px;")
        info_label.setWordWrap(True)
        main_layout.addWidget(info_label)

        # 圖像顯示區域（帶滾動條）
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("background-color: #555555;")

        self.image_label = ImageLabel()
        self.image_label.setMinimumSize(800, 600)
        scroll_area.setWidget(self.image_label)

        main_layout.addWidget(scroll_area)

        # 狀態列
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("請開啟圖像")

    def load_image(self):
        """載入圖像"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "選擇圖像",
            "",
            "圖像檔案 (*.png *.jpg *.jpeg *.bmp *.gif);;所有檔案 (*.*)",
        )

        if not file_path:
            return

        if self.processor.load_image(file_path):
            self.current_image = self.processor.image.copy()
            self.display_image(self.current_image)
            self.status_bar.showMessage(f"已載入：{Path(file_path).name}")
            self.image_label.clear_selection()
            self.detected_mask = None
        else:
            QMessageBox.critical(self, "錯誤", "無法載入圖像")

    def display_image(self, image: np.ndarray):
        """在標籤上顯示圖像"""
        # 轉換 BGR 到 RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 取得圖像尺寸
        height, width, channels = image_rgb.shape
        bytes_per_line = channels * width

        # 建立 QImage
        q_image = QImage(
            image_rgb.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        )

        # 轉換為 QPixmap 並顯示
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap)
        self.image_label.adjustSize()

    def auto_detect(self):
        """自動偵測浮水印"""
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "請先載入圖像")
            return

        mask = self.processor.auto_detect_watermark()

        if mask is None:
            QMessageBox.information(self, "資訊", "未偵測到浮水印")
            return

        # 在圖像上顯示偵測結果
        display_image = self.current_image.copy()
        # 將遮罩區域標記為紅色
        display_image[mask > 0] = [0, 0, 255]

        self.display_image(display_image)
        self.status_bar.showMessage("已自動偵測浮水印區域（紅色標記）")

        # 儲存遮罩以供後續移除使用
        self.detected_mask = mask
        self.image_label.clear_selection()

    def remove_watermark(self):
        """移除浮水印"""
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "請先載入圖像")
            return

        try:
            # 檢查是否有自動偵測的遮罩
            if self.detected_mask is not None:
                result = self.processor.remove_watermark(self.detected_mask)
                self.detected_mask = None
            else:
                # 使用手動選擇的區域
                selection = self.image_label.get_selection_coords()
                if selection:
                    x, y, width, height = selection
                    result = self.processor.remove_watermark_by_region(
                        x, y, width, height
                    )
                else:
                    QMessageBox.warning(self, "警告", "請先選擇浮水印區域或使用自動偵測")
                    return

            self.current_image = result
            self.processor.image = result.copy()
            self.display_image(result)
            self.status_bar.showMessage("已移除浮水印")

            # 清除選擇
            self.image_label.clear_selection()

        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"移除浮水印時發生錯誤：{str(e)}")

    def save_image(self):
        """儲存圖像"""
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "沒有可儲存的圖像")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "儲存圖像",
            "",
            "PNG 檔案 (*.png);;JPEG 檔案 (*.jpg);;所有檔案 (*.*)",
        )

        if not file_path:
            return

        if self.processor.save_image(self.current_image, file_path):
            QMessageBox.information(self, "成功", f"圖像已儲存至：{file_path}")
            self.status_bar.showMessage(f"已儲存：{Path(file_path).name}")
        else:
            QMessageBox.critical(self, "錯誤", "儲存圖像失敗")

    def reset_image(self):
        """重置為原始圖像"""
        if self.processor.original is None:
            return

        self.current_image = self.processor.original.copy()
        self.processor.image = self.current_image.copy()
        self.display_image(self.current_image)
        self.status_bar.showMessage("已重置為原始圖像")

        # 清除選擇
        self.image_label.clear_selection()
        self.detected_mask = None


def run_gui():
    """啟動 GUI 應用程式"""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # 使用 Fusion 風格

    window = WatermarkRemoverGUI()
    window.show()

    sys.exit(app.exec())
