"""增強版 GUI 介面模組 - 使用 PySide6"""

from pathlib import Path
import sys
from collections import deque

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
    QProgressDialog,
    QSlider,
    QToolBar,
    QMenuBar,
    QComboBox,
)
from PySide6.QtCore import Qt, QRect, QPoint, QThread, Signal, QMimeData
from PySide6.QtGui import (
    QImage,
    QPixmap,
    QPainter,
    QPen,
    QColor,
    QAction,
    QKeySequence,
    QDragEnterEvent,
    QDropEvent,
)

from .image_processor import ImageProcessor, InpaintMethod
from .model_manager import ModelManager


class ProcessingThread(QThread):
    """處理圖像的背景執行緒"""

    finished = Signal(np.ndarray)
    error = Signal(str)
    progress = Signal(int)
    fallback = Signal(str)

    def __init__(
        self,
        processor: ImageProcessor,
        mask: np.ndarray | None,
        selection: tuple[int, int, int, int] | None,
        method: InpaintMethod = None,
    ):
        super().__init__()
        self.processor = processor
        self.mask = mask
        self.selection = selection
        self.method = method

    def run(self):
        """執行處理"""
        try:
            self.progress.emit(10)

            def progress_cb(value):
                self.progress.emit(value)

            def fallback_cb(msg):
                self.fallback.emit(msg)

            if self.mask is not None:
                result = self.processor.remove_watermark(
                    self.mask,
                    method=self.method,
                    progress_callback=progress_cb,
                    fallback_callback=fallback_cb,
                )
            elif self.selection:
                x, y, width, height = self.selection
                result = self.processor.remove_watermark_by_region(
                    x, y, width, height,
                    method=self.method,
                    progress_callback=progress_cb,
                    fallback_callback=fallback_cb,
                )
            else:
                self.error.emit("未選擇區域")
                return

            self.finished.emit(result)

        except Exception as e:
            self.error.emit(str(e))


class ModelDownloadThread(QThread):
    """下載模型的背景執行緒"""

    finished = Signal()
    error = Signal(str)
    progress = Signal(int, int)  # downloaded, total

    def __init__(self, model_manager):
        super().__init__()
        self.model_manager = model_manager

    def run(self):
        """執行下載"""
        try:
            self.model_manager.download_model(
                "lama",
                progress_callback=lambda downloaded, total: self.progress.emit(downloaded, total),
            )
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


class ZoomableImageLabel(QLabel):
    """可縮放和繪製選擇區域的圖像標籤"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScaledContents(False)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setAcceptDrops(True)

        self.selecting = False
        self.start_point = QPoint()
        self.end_point = QPoint()
        self.selection_rect: QRect | None = None

        # 縮放相關
        self.zoom_level = 1.0
        self.original_pixmap: QPixmap | None = None

    def set_image(self, pixmap: QPixmap):
        """設定圖像"""
        self.original_pixmap = pixmap
        self.apply_zoom()

    def apply_zoom(self):
        """應用縮放"""
        if self.original_pixmap:
            if self.zoom_level == 1.0:
                self.setPixmap(self.original_pixmap)
            else:
                scaled_pixmap = self.original_pixmap.scaled(
                    int(self.original_pixmap.width() * self.zoom_level),
                    int(self.original_pixmap.height() * self.zoom_level),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self.setPixmap(scaled_pixmap)
            self.adjustSize()

    def zoom_in(self):
        """放大"""
        self.zoom_level = min(self.zoom_level * 1.2, 5.0)
        self.apply_zoom()

    def zoom_out(self):
        """縮小"""
        self.zoom_level = max(self.zoom_level / 1.2, 0.2)
        self.apply_zoom()

    def zoom_reset(self):
        """重置縮放"""
        self.zoom_level = 1.0
        self.apply_zoom()

    def wheelEvent(self, event):
        """滑鼠滾輪事件 - 縮放"""
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if event.angleDelta().y() > 0:
                self.zoom_in()
            else:
                self.zoom_out()
            event.accept()
        else:
            super().wheelEvent(event)

    def dragEnterEvent(self, event: QDragEnterEvent):
        """拖曳進入事件"""
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        """放下事件"""
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            # 通知父視窗載入圖像
            parent = self.parent()
            while parent and not isinstance(parent, WatermarkRemoverGUI):
                parent = parent.parent()
            if parent:
                parent.load_image_from_path(file_path)

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

            # 計算選擇區域（考慮縮放）
            x1 = int(min(self.start_point.x(), self.end_point.x()) / self.zoom_level)
            y1 = int(min(self.start_point.y(), self.end_point.y()) / self.zoom_level)
            x2 = int(max(self.start_point.x(), self.end_point.x()) / self.zoom_level)
            y2 = int(max(self.start_point.y(), self.end_point.y()) / self.zoom_level)

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

        # 繪製當前拖曳的矩形（考慮縮放）
        if self.selecting:
            rect = QRect(self.start_point, self.end_point).normalized()
            painter.drawRect(rect)

        # 繪製已確認的選擇區域（考慮縮放）
        elif self.selection_rect:
            scaled_rect = QRect(
                int(self.selection_rect.x() * self.zoom_level),
                int(self.selection_rect.y() * self.zoom_level),
                int(self.selection_rect.width() * self.zoom_level),
                int(self.selection_rect.height() * self.zoom_level),
            )
            painter.drawRect(scaled_rect)

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
    """增強版浮水印移除工具的主視窗"""

    def __init__(self):
        super().__init__()
        self.processor = ImageProcessor()
        self.model_manager = ModelManager()
        self.current_image: np.ndarray | None = None
        self.detected_mask: np.ndarray | None = None
        self.lama_available = False
        self.current_method = InpaintMethod.LAMA

        # 歷史記錄（用於撤銷/重做）
        self.history: deque = deque(maxlen=10)
        self.redo_stack: list = []

        self.setup_ui()
        self.setup_shortcuts()

        # 啟動時檢查模型
        self.check_model_on_startup()

    def setup_ui(self):
        """設定 UI 元件"""
        self.setWindowTitle("圖像浮水印移除工具 (增強版)")
        self.setGeometry(100, 100, 1400, 900)

        # 建立選單列
        self.create_menu_bar()

        # 建立工具列
        self.create_toolbar()

        # 中央 widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主佈局
        main_layout = QVBoxLayout(central_widget)

        # 說明文字
        info_label = QLabel(
            "使用方式：開啟圖像 (Ctrl+O) → 選擇浮水印區域 → 移除 (Ctrl+R) → 儲存 (Ctrl+S) | 提示：可拖放檔案、Ctrl+滾輪縮放"
        )
        info_label.setStyleSheet("color: #2196F3; padding: 8px; font-size: 11px;")
        info_label.setWordWrap(True)
        main_layout.addWidget(info_label)

        # 圖像顯示區域（帶滾動條）
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("background-color: #424242;")

        self.image_label = ZoomableImageLabel()
        self.image_label.setMinimumSize(800, 600)
        scroll_area.setWidget(self.image_label)

        main_layout.addWidget(scroll_area)

        # 縮放控制列
        zoom_layout = QHBoxLayout()
        zoom_layout.addWidget(QLabel("縮放:"))

        self.btn_zoom_out = QPushButton("−")
        self.btn_zoom_out.setMaximumWidth(40)
        self.btn_zoom_out.clicked.connect(self.image_label.zoom_out)
        zoom_layout.addWidget(self.btn_zoom_out)

        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setMinimum(20)
        self.zoom_slider.setMaximum(500)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setMaximumWidth(200)
        self.zoom_slider.valueChanged.connect(self.on_zoom_slider_changed)
        zoom_layout.addWidget(self.zoom_slider)

        self.btn_zoom_in = QPushButton("+")
        self.btn_zoom_in.setMaximumWidth(40)
        self.btn_zoom_in.clicked.connect(self.image_label.zoom_in)
        zoom_layout.addWidget(self.btn_zoom_in)

        self.zoom_label = QLabel("100%")
        self.zoom_label.setMinimumWidth(50)
        zoom_layout.addWidget(self.zoom_label)

        zoom_layout.addStretch()

        main_layout.addLayout(zoom_layout)

        # 算法選擇列
        algorithm_layout = QHBoxLayout()
        algorithm_layout.addWidget(QLabel("修復算法:"))

        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems([
            "LaMa (AI) - 推薦，適合大面積",
            "NS - 快速，適合小面積",
            "Telea - 最快",
            "Hybrid - 混合",
        ])
        self.algorithm_combo.setCurrentIndex(0)
        self.algorithm_combo.currentIndexChanged.connect(self.on_algorithm_changed)
        self.algorithm_combo.setMinimumWidth(250)
        algorithm_layout.addWidget(self.algorithm_combo)

        algorithm_layout.addStretch()

        main_layout.addLayout(algorithm_layout)

        # 狀態列
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("請開啟圖像或拖放圖像檔案到視窗中")

    def create_menu_bar(self):
        """建立選單列"""
        menubar = self.menuBar()

        # 檔案選單
        file_menu = menubar.addMenu("檔案(&F)")

        open_action = QAction("開啟圖像(&O)", self)
        open_action.setShortcut(QKeySequence("Ctrl+O"))
        open_action.triggered.connect(self.load_image)
        file_menu.addAction(open_action)

        save_action = QAction("儲存圖像(&S)", self)
        save_action.setShortcut(QKeySequence("Ctrl+S"))
        save_action.triggered.connect(self.save_image)
        file_menu.addAction(save_action)

        save_as_action = QAction("另存新檔(&A)", self)
        save_as_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        save_as_action.triggered.connect(self.save_image)
        file_menu.addAction(save_as_action)

        file_menu.addSeparator()

        exit_action = QAction("退出(&X)", self)
        exit_action.setShortcut(QKeySequence("Ctrl+Q"))
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # 編輯選單
        edit_menu = menubar.addMenu("編輯(&E)")

        self.undo_action = QAction("撤銷(&U)", self)
        self.undo_action.setShortcut(QKeySequence("Ctrl+Z"))
        self.undo_action.triggered.connect(self.undo)
        self.undo_action.setEnabled(False)
        edit_menu.addAction(self.undo_action)

        self.redo_action = QAction("重做(&R)", self)
        self.redo_action.setShortcut(QKeySequence("Ctrl+Y"))
        self.redo_action.triggered.connect(self.redo)
        self.redo_action.setEnabled(False)
        edit_menu.addAction(self.redo_action)

        edit_menu.addSeparator()

        reset_action = QAction("重置圖像(&T)", self)
        reset_action.setShortcut(QKeySequence("Ctrl+T"))
        reset_action.triggered.connect(self.reset_image)
        edit_menu.addAction(reset_action)

        # 檢視選單
        view_menu = menubar.addMenu("檢視(&V)")

        zoom_in_action = QAction("放大(&I)", self)
        zoom_in_action.setShortcut(QKeySequence("Ctrl++"))
        zoom_in_action.triggered.connect(self.zoom_in)
        view_menu.addAction(zoom_in_action)

        zoom_out_action = QAction("縮小(&O)", self)
        zoom_out_action.setShortcut(QKeySequence("Ctrl+-"))
        zoom_out_action.triggered.connect(self.zoom_out)
        view_menu.addAction(zoom_out_action)

        zoom_reset_action = QAction("重置縮放(&R)", self)
        zoom_reset_action.setShortcut(QKeySequence("Ctrl+0"))
        zoom_reset_action.triggered.connect(self.zoom_reset)
        view_menu.addAction(zoom_reset_action)

        # 工具選單
        tools_menu = menubar.addMenu("工具(&T)")

        auto_detect_action = QAction("自動偵測浮水印(&A)", self)
        auto_detect_action.setShortcut(QKeySequence("Ctrl+D"))
        auto_detect_action.triggered.connect(self.auto_detect)
        tools_menu.addAction(auto_detect_action)

        remove_action = QAction("移除浮水印(&R)", self)
        remove_action.setShortcut(QKeySequence("Ctrl+R"))
        remove_action.triggered.connect(self.remove_watermark)
        tools_menu.addAction(remove_action)

    def create_toolbar(self):
        """建立工具列"""
        toolbar = QToolBar("主工具列")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # 開啟按鈕
        self.btn_open = QAction("📂 開啟", self)
        self.btn_open.triggered.connect(self.load_image)
        toolbar.addAction(self.btn_open)

        toolbar.addSeparator()

        # 自動偵測按鈕
        self.btn_auto_detect = QAction("🔍 自動偵測", self)
        self.btn_auto_detect.triggered.connect(self.auto_detect)
        toolbar.addAction(self.btn_auto_detect)

        # 移除浮水印按鈕
        self.btn_remove = QAction("🔧 移除浮水印", self)
        self.btn_remove.triggered.connect(self.remove_watermark)
        toolbar.addAction(self.btn_remove)

        toolbar.addSeparator()

        # 撤銷/重做按鈕
        self.toolbar_undo = QAction("↶ 撤銷", self)
        self.toolbar_undo.triggered.connect(self.undo)
        self.toolbar_undo.setEnabled(False)
        toolbar.addAction(self.toolbar_undo)

        self.toolbar_redo = QAction("↷ 重做", self)
        self.toolbar_redo.triggered.connect(self.redo)
        self.toolbar_redo.setEnabled(False)
        toolbar.addAction(self.toolbar_redo)

        toolbar.addSeparator()

        # 儲存按鈕
        self.btn_save = QAction("💾 儲存", self)
        self.btn_save.triggered.connect(self.save_image)
        toolbar.addAction(self.btn_save)

        # 重置按鈕
        self.btn_reset = QAction("↺ 重置", self)
        self.btn_reset.triggered.connect(self.reset_image)
        toolbar.addAction(self.btn_reset)

    def setup_shortcuts(self):
        """設定快捷鍵（額外的快捷鍵）"""
        pass  # 主要快捷鍵已在選單中設定

    def on_zoom_slider_changed(self, value):
        """縮放滑桿改變"""
        zoom = value / 100.0
        self.image_label.zoom_level = zoom
        self.image_label.apply_zoom()
        self.zoom_label.setText(f"{value}%")

    def zoom_in(self):
        """放大"""
        self.image_label.zoom_in()
        self.update_zoom_controls()

    def zoom_out(self):
        """縮小"""
        self.image_label.zoom_out()
        self.update_zoom_controls()

    def zoom_reset(self):
        """重置縮放"""
        self.image_label.zoom_reset()
        self.update_zoom_controls()

    def update_zoom_controls(self):
        """更新縮放控制"""
        value = int(self.image_label.zoom_level * 100)
        self.zoom_slider.setValue(value)
        self.zoom_label.setText(f"{value}%")

    def add_to_history(self):
        """添加到歷史記錄"""
        if self.current_image is not None:
            self.history.append(self.current_image.copy())
            self.redo_stack.clear()
            self.update_undo_redo_state()

    def undo(self):
        """撤銷"""
        if len(self.history) > 1:
            # 當前圖像加入重做堆疊
            self.redo_stack.append(self.current_image.copy())
            # 從歷史中取出上一個圖像
            self.history.pop()
            self.current_image = self.history[-1].copy()
            self.processor.image = self.current_image.copy()
            self.display_image(self.current_image)
            self.status_bar.showMessage("已撤銷")
            self.update_undo_redo_state()

    def redo(self):
        """重做"""
        if self.redo_stack:
            # 當前圖像加入歷史
            self.history.append(self.current_image.copy())
            # 從重做堆疊取出圖像
            self.current_image = self.redo_stack.pop().copy()
            self.processor.image = self.current_image.copy()
            self.display_image(self.current_image)
            self.status_bar.showMessage("已重做")
            self.update_undo_redo_state()

    def update_undo_redo_state(self):
        """更新撤銷/重做按鈕狀態"""
        can_undo = len(self.history) > 1
        can_redo = len(self.redo_stack) > 0

        self.undo_action.setEnabled(can_undo)
        self.toolbar_undo.setEnabled(can_undo)

        self.redo_action.setEnabled(can_redo)
        self.toolbar_redo.setEnabled(can_redo)

    def load_image_from_path(self, file_path: str):
        """從路徑載入圖像（用於拖放）"""
        if self.processor.load_image(file_path):
            self.current_image = self.processor.image.copy()
            self.display_image(self.current_image)
            self.status_bar.showMessage(f"已載入：{Path(file_path).name}")
            self.image_label.clear_selection()
            self.detected_mask = None
            self.history.clear()
            self.redo_stack.clear()
            self.add_to_history()
        else:
            QMessageBox.critical(self, "錯誤", "無法載入圖像")

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

        self.load_image_from_path(file_path)

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
        self.image_label.set_image(pixmap)

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

    def on_algorithm_changed(self, index):
        """算法選擇改變"""
        methods = [InpaintMethod.LAMA, InpaintMethod.NS, InpaintMethod.TELEA, InpaintMethod.HYBRID]
        self.current_method = methods[index]
        self.processor.default_method = self.current_method
        self.status_bar.showMessage(f"已切換至 {self.algorithm_combo.currentText().split(' - ')[0]} 算法")

    def check_model_on_startup(self):
        """啟動時檢查模型"""
        if self.model_manager.is_model_downloaded("lama"):
            self.lama_available = True
            self.status_bar.showMessage("LaMa 模型已就緒，請開啟圖像")
        else:
            reply = QMessageBox.question(
                self,
                "下載 AI 模型",
                "LaMa AI 模型尚未下載（約 200 MB）。\n"
                "此模型可大幅提升大面積浮水印的移除品質。\n\n"
                "是否立即下載？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )

            if reply == QMessageBox.StandardButton.Yes:
                self.download_model_with_progress()
            else:
                self.lama_available = False
                self.update_algorithm_menu()
                self.status_bar.showMessage("請開啟圖像或拖放圖像檔案到視窗中")

    def download_model_with_progress(self):
        """顯示下載進度"""
        self.download_progress = QProgressDialog(
            "正在下載 LaMa 模型...\n這可能需要幾分鐘",
            "取消",
            0,
            100,
            self,
        )
        self.download_progress.setWindowTitle("下載模型")
        self.download_progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.download_progress.setMinimumDuration(0)
        self.download_progress.setValue(0)

        self.download_thread = ModelDownloadThread(self.model_manager)
        self.download_thread.progress.connect(self.on_download_progress)
        self.download_thread.finished.connect(self.on_download_finished)
        self.download_thread.error.connect(self.on_download_error)
        self.download_thread.start()

    def on_download_progress(self, downloaded: int, total: int):
        """下載進度更新"""
        if total > 0:
            percent = int(downloaded / total * 100)
            self.download_progress.setValue(percent)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total / (1024 * 1024)
            self.download_progress.setLabelText(
                f"正在下載 LaMa 模型...\n{mb_downloaded:.1f} / {mb_total:.1f} MB"
            )

    def on_download_finished(self):
        """下載完成"""
        self.download_progress.close()
        self.lama_available = True
        self.update_algorithm_menu()
        QMessageBox.information(self, "成功", "LaMa 模型下載完成！")
        self.status_bar.showMessage("LaMa 模型已就緒，請開啟圖像")

    def on_download_error(self, error: str):
        """下載錯誤"""
        self.download_progress.close()
        QMessageBox.critical(self, "錯誤", f"下載模型失敗：{error}")
        self.lama_available = False
        self.update_algorithm_menu()

    def update_algorithm_menu(self):
        """更新算法選單狀態"""
        if not self.lama_available:
            model = self.algorithm_combo.model()
            item = model.item(0)
            item.setEnabled(False)
            self.algorithm_combo.setItemText(0, "LaMa (AI) - 未下載")
            self.algorithm_combo.setCurrentIndex(1)
            self.current_method = InpaintMethod.NS
            self.processor.default_method = InpaintMethod.NS
        else:
            model = self.algorithm_combo.model()
            item = model.item(0)
            item.setEnabled(True)
            self.algorithm_combo.setItemText(0, "LaMa (AI) - 推薦，適合大面積")

    def remove_watermark(self):
        """移除浮水印（使用進度對話框）"""
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "請先載入圖像")
            return

        selection = self.image_label.get_selection_coords()

        if self.detected_mask is None and not selection:
            QMessageBox.warning(self, "警告", "請先選擇浮水印區域或使用自動偵測")
            return

        # 先添加到歷史記錄
        self.add_to_history()

        # 建立進度對話框
        progress = QProgressDialog("正在移除浮水印...", "取消", 0, 100, self)
        progress.setWindowTitle("處理中")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        # 建立處理執行緒
        self.processing_thread = ProcessingThread(
            self.processor,
            self.detected_mask,
            selection,
            self.current_method,  # 傳遞當前選擇的算法
        )

        self.processing_thread.progress.connect(progress.setValue)
        self.processing_thread.finished.connect(
            lambda result: self.on_processing_finished(result, progress)
        )
        self.processing_thread.error.connect(
            lambda error: self.on_processing_error(error, progress)
        )
        self.processing_thread.fallback.connect(
            lambda msg: self.status_bar.showMessage(msg)
        )

        self.processing_thread.start()

    def on_processing_finished(self, result: np.ndarray, progress: QProgressDialog):
        """處理完成"""
        progress.close()

        self.current_image = result
        self.processor.image = result.copy()
        self.display_image(result)
        self.status_bar.showMessage("已移除浮水印")

        # 清除選擇和遮罩
        self.image_label.clear_selection()
        self.detected_mask = None

    def on_processing_error(self, error: str, progress: QProgressDialog):
        """處理錯誤"""
        progress.close()
        QMessageBox.critical(self, "錯誤", f"移除浮水印時發生錯誤：{error}")

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

        # 清除選擇、歷史和遮罩
        self.image_label.clear_selection()
        self.detected_mask = None
        self.history.clear()
        self.redo_stack.clear()
        self.add_to_history()


def run_gui():
    """啟動增強版 GUI 應用程式"""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # 使用 Fusion 風格

    window = WatermarkRemoverGUI()
    window.show()

    sys.exit(app.exec())
