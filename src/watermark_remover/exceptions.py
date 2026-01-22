"""LaMa 相關例外類別"""


class LamaError(Exception):
    """LaMa 相關錯誤基類"""
    pass


class LamaModelNotFoundError(LamaError):
    """模型檔案不存在"""
    pass


class LamaDownloadError(LamaError):
    """模型下載失敗"""
    pass


class LamaInferenceError(LamaError):
    """推論過程錯誤"""
    pass


class LamaMemoryError(LamaError):
    """記憶體不足"""
    pass
