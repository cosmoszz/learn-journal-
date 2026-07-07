"""
utils 包初始化文件
"""

# 导入主要模块，方便使用
from .config import config
from .modelscope_client import modelscope_client, get_client

__all__ = ['config', 'modelscope_client', 'get_client']