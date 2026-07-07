"""
配置管理模块
统一管理项目中的各种配置和环境变量
"""

import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

class Config:
    """配置类，集中管理所有配置项"""
    
    # ModelScope 配置
    MODELSCOPE_API_KEY = os.getenv('MODELSCOPE_API_KEY')
    MODELSCOPE_BASE_URL = os.getenv('MODELSCOPE_BASE_URL', 'https://api-inference.modelscope.cn/v1')
    
    # OpenAI 兼容配置（如果以后需要）
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
    
    @classmethod
    def validate_modelscope_config(cls):
        """验证 ModelScope 配置是否完整"""
        if not cls.MODELSCOPE_API_KEY:
            raise ValueError("MODELSCOPE_API_KEY 未在环境变量中设置")
        if not cls.MODELSCOPE_BASE_URL:
            raise ValueError("MODELSCOPE_BASE_URL 未在环境变量中设置")
        return True

# 默认配置实例
config = Config()