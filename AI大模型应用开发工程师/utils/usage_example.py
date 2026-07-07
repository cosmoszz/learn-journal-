"""
使用示例：如何在项目中使用公共的 ModelScope 客户端

只需简单导入即可使用，无需重复初始化代码
"""

# 方法1：直接使用预创建的客户端实例
from LearnAI.utils.modelscope_client import modelscope_client,get_message,print_response

# 现在你可以直接使用 modelscope_client 来调用 API
# response = modelscope_client.chat.completions.create(
#     model="inclusionAI/Ling-2.6-1T",  # 示例模型名称，根据实际可用模型调整
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "你好，请介绍一下自己"}
#     ],
#     temperature=0.7,
#     max_tokens=500
# )

review = '这款音效特别不好 给你意想不到的音质。'
messages=[
    {"role": "system", "content": "你是一名舆情分析师，帮我判断产品口碑的正负向，回复请用一个词语：正向 或者 负向"},
    {"role": "user", "content": review}
  ]
messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你好，请介绍一下自己"}
    ]
# response = get_message("用户: 你好，请介绍一下自己",model="inclusionAI/Ling-2.6-1T",isStream= True, temperature=0.7,max_tokens=500)
response = get_message(messages,
                       model="deepseek-ai/DeepSeek-V4-Flash",isStream= True,
                       temperature=0.7,max_tokens=1000000
                       )

# print(response.choices[0].message.content)

print_response(response)

# 方法2：使用便捷函数获取客户端
from LearnAI.utils.modelscope_client import get_client

client = get_client()
# 使用 client 进行 API 调用


# 方法3：如果你需要自定义配置，也可以直接使用配置
from LearnAI.utils.config import config

print(f"Base URL: {config.MODELSCOPE_BASE_URL}")
print(f"API Key 存在: {bool(config.MODELSCOPE_API_KEY)}")