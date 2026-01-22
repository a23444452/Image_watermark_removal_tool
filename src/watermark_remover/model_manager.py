"""模型管理器：負責 AI 模型的下載和管理"""

from pathlib import Path
from typing import Callable
import urllib.request
import shutil

from .exceptions import LamaDownloadError, LamaModelNotFoundError


class ModelManager:
    """管理 AI 模型的下載與儲存"""

    MODEL_DIR = Path.home() / ".cache" / "watermark-remover" / "models"

    # LaMa 模型資訊
    MODELS = {
        "lama": {
            "filename": "big-lama.pt",
            "url": "https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.pt",
            "size": 200 * 1024 * 1024,  # 約 200 MB
        }
    }

    def __init__(self):
        self.model_dir = self.MODEL_DIR
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def is_model_downloaded(self, model_name: str = "lama") -> bool:
        """檢查模型是否已下載"""
        model_path = self.get_model_path(model_name)
        return model_path.exists()

    def get_model_path(self, model_name: str = "lama") -> Path:
        """取得模型路徑"""
        if model_name not in self.MODELS:
            raise ValueError(f"未知的模型: {model_name}")

        model_info = self.MODELS[model_name]
        return self.model_dir / model_name / model_info["filename"]

    def get_model_size(self, model_name: str = "lama") -> int:
        """取得模型預估大小（bytes）"""
        if model_name not in self.MODELS:
            raise ValueError(f"未知的模型: {model_name}")
        return self.MODELS[model_name]["size"]

    def download_model(
        self,
        model_name: str = "lama",
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> Path:
        """
        下載模型

        Args:
            model_name: 模型名稱
            progress_callback: 進度回調函數 (downloaded_bytes, total_bytes)

        Returns:
            下載完成的模型路徑

        Raises:
            LamaDownloadError: 下載失敗
        """
        if model_name not in self.MODELS:
            raise ValueError(f"未知的模型: {model_name}")

        model_info = self.MODELS[model_name]
        model_path = self.get_model_path(model_name)

        # 建立模型目錄
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # 下載模型
        try:
            temp_path = model_path.with_suffix(".tmp")

            # 使用 urllib 下載
            with urllib.request.urlopen(model_info["url"]) as response:
                total_size = int(response.headers.get("Content-Length", 0))
                downloaded = 0
                chunk_size = 8192

                with open(temp_path, "wb") as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)

                        if progress_callback:
                            progress_callback(downloaded, total_size)

            # 下載完成，重命名檔案
            shutil.move(str(temp_path), str(model_path))

            return model_path

        except Exception as e:
            # 清理臨時檔案
            temp_path = model_path.with_suffix(".tmp")
            if temp_path.exists():
                temp_path.unlink()
            raise LamaDownloadError(f"下載模型失敗: {e}") from e

    def delete_model(self, model_name: str = "lama") -> bool:
        """刪除已下載的模型"""
        model_path = self.get_model_path(model_name)
        if model_path.exists():
            model_path.unlink()
            return True
        return False
