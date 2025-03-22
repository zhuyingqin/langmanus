"""
日志处理模块，用于配置日志输出和保存日志到文件。
"""

import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

# 默认日志目录
DEFAULT_LOG_DIR = "logs"

def setup_logging(log_dir=DEFAULT_LOG_DIR, debug=False, save_to_file=True):
    """
    设置日志系统，可以选择是否将日志保存到文件。
    
    Args:
        log_dir: 日志存储目录
        debug: 是否启用调试级别日志
        save_to_file: 是否将日志保存到文件
    """
    # 创建日志目录（如果不存在）
    if save_to_file and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 获取根日志记录器
    root_logger = logging.getLogger()
    
    # 设置日志级别
    level = logging.DEBUG if debug else logging.INFO
    root_logger.setLevel(level)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # 设置格式
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    
    # 清除已有的处理器
    if root_logger.handlers:
        root_logger.handlers.clear()
    
    # 添加控制台处理器
    root_logger.addHandler(console_handler)
    
    # 如果启用文件日志
    if save_to_file:
        # 创建带日期的日志文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"langmanus_{timestamp}.log"
        log_path = os.path.join(log_dir, log_filename)
        
        # 创建文件处理器，最大10MB，保留5个备份
        file_handler = RotatingFileHandler(
            log_path, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        
        # 添加文件处理器
        root_logger.addHandler(file_handler)
        
        # 创建专门用于调试信息的文件处理器
        if debug:
            debug_log_path = os.path.join(log_dir, f"debug_{timestamp}.log")
            debug_handler = RotatingFileHandler(
                debug_log_path, maxBytes=20*1024*1024, backupCount=3, encoding='utf-8'
            )
            debug_handler.setLevel(logging.DEBUG)
            debug_handler.setFormatter(formatter)
            
            # 添加一个过滤器，只保存DEBUG级别的消息
            class DebugLevelFilter(logging.Filter):
                def filter(self, record):
                    return record.levelno == logging.DEBUG
                
            debug_handler.addFilter(DebugLevelFilter())
            root_logger.addHandler(debug_handler)
            
            logging.info(f"调试日志已启用，将保存到 {debug_log_path}")
        
        logging.info(f"日志将保存到 {log_path}")
    
    return root_logger

def enable_debug_logging():
    """启用调试级别日志"""
    logging.getLogger("src").setLevel(logging.DEBUG)
    logging.getLogger().setLevel(logging.DEBUG)
    logging.info("已启用调试级别日志") 