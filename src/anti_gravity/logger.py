import logging
import sys
from pathlib import Path
from anti_gravity.settings import settings

def setup_logger(name: str) -> logging.Logger:
    """
    🟣 除錯與穩定性：Structured Logging
    配置統一的日誌記錄器，支援 Console 與 File 輸出。
    """
    logger = logging.getLogger(name)
    
    # 取得設定好的 Level
    level = getattr(logging, settings.logging.level.upper(), logging.INFO)
    logger.setLevel(level)
    
    # 防止多重實例化的 Handler 累積
    if logger.hasHandlers():
        return logger

    formatter = logging.Formatter(settings.logging.format)

    # 1. Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 2. File Handler (如果設定中有路徑)
    if settings.logging.log_file:
        log_file = Path(settings.logging.log_file)
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            # 即使檔案無法寫入，也要確保 Console 還有日誌
            print(f"⚠️ [Logger] 無法建立日誌檔案 {log_file}: {e}")

    return logger

# 🚀 導出預設 logger
logger = setup_logger("anti_gravity")
