"""ModelManager 測試"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import os

from watermark_remover.model_manager import ModelManager


class TestModelManager:
    """ModelManager 測試類"""

    def test_model_dir_created_on_init(self):
        """測試初始化時建立模型目錄"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(ModelManager, 'MODEL_DIR', Path(tmpdir) / "models"):
                manager = ModelManager()
                assert manager.model_dir.exists()

    def test_is_model_downloaded_false_when_not_exists(self):
        """測試模型未下載時返回 False"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(ModelManager, 'MODEL_DIR', Path(tmpdir) / "models"):
                manager = ModelManager()
                assert manager.is_model_downloaded("lama") is False

    def test_is_model_downloaded_true_when_exists(self):
        """測試模型存在時返回 True"""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "models" / "lama"
            model_dir.mkdir(parents=True)
            (model_dir / "big-lama.pt").touch()

            with patch.object(ModelManager, 'MODEL_DIR', Path(tmpdir) / "models"):
                manager = ModelManager()
                assert manager.is_model_downloaded("lama") is True

    def test_get_model_path(self):
        """測試取得模型路徑"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(ModelManager, 'MODEL_DIR', Path(tmpdir) / "models"):
                manager = ModelManager()
                path = manager.get_model_path("lama")
                assert path == Path(tmpdir) / "models" / "lama" / "big-lama.pt"
