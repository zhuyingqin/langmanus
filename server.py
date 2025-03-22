"""
Server script for running the LangManus API.
"""

import logging
import uvicorn
import sys
from src.utils.log_handler import setup_logging

# 设置日志系统，默认保存日志到文件
setup_logging(debug=False, save_to_file=True)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting LangManus API server")
    reload = True
    if sys.platform.startswith("win"):
        reload = False
    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=reload,
        log_level="info",
    )
