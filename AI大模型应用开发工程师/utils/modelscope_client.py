"""
ModelScope 客户端初始化模块
提供统一的 OpenAI 兼容客户端实例
"""

from openai import OpenAI
from .config import config


def create_modelscope_client():
    """
    创建 ModelScope 的 OpenAI 兼容客户端
    :return: 配置好的 OpenAI 客户端实例
    """
    # 验证配置
    config.validate_modelscope_config()
    
    client = OpenAI(
        base_url=config.MODELSCOPE_BASE_URL,
        api_key=config.MODELSCOPE_API_KEY,
    )
    
    return client


def get_message(content, model='deepseek-ai/DeepSeek-V4-Flash',isStream=True,
                **kwargs):
    """
    获取模型回复的便捷函数
    :param content: 消息内容
    :param model: 模型名称，默认为 'deepseek-ai/DeepSeek-V4-Flash'
    :param temperature: 温度参数，默认0.7
    :param max_tokens: 最大token数，默认500
    :param isStream: 是否流式返回，默认True
    :param kwargs: 其他传递给API的参数
    :return: API响应结果
    """
    params = {
        "model": model,
        "messages": content,
        "stream": isStream,
    }
    # 更新其他参数
    params.update(kwargs)

    response = get_client().chat.completions.create(**params)
    return response


def print_response(response):
    done_reasoning = False  # 标记思考是否结束
    is_reasoning_started = False  # 标记思考是否开始

    for chunk in response:
        if chunk.choices:
            # 使用 getattr 安全取值，防止报错
            reasoning_chunk = getattr(chunk.choices[0].delta, 'reasoning_content', None) or ''
            answer_chunk = chunk.choices[0].delta.content or ''

            # 1. 处理思考过程
            if reasoning_chunk != '':
                # 如果是第一次收到思考内容，先打印 <think> 开头
                if not is_reasoning_started:
                    print('<think>', end='', flush=True)
                    is_reasoning_started = True
                # 打印思考片段
                print(reasoning_chunk, end='', flush=True)

            # 2. 处理最终回答
            elif answer_chunk != '':
                # 如果是第一次收到回答，说明思考结束了
                if not done_reasoning:
                    # 如果之前有思考过程，闭合 </think> 标签
                    if is_reasoning_started:
                        print('</think>\n\n', end='', flush=True)
                    done_reasoning = True

                # 打印回答片段
                print(answer_chunk, end='', flush=True)


# 创建全局客户端实例
modelscope_client = create_modelscope_client()


def get_client():
    """
    获取 ModelScope 客户端实例的便捷函数
    :return: OpenAI 客户端实例
    """
    return modelscope_client