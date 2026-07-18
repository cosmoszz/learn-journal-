import sys
import io
import os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from openai import OpenAI
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

client = OpenAI(
    base_url=os.getenv('MODELSCOPE_BASE_URL'),
    api_key=os.getenv('MODELSCOPE_API_KEY'),
)

# 是否流式返回
isStream=True
# isStream=False
# 舆情分析：情感分析示例
review = '这款音效特别好 给你意想不到的音质。'

messages = [
    {'role': 'system', 'content': '你是一名舆情分析师，帮我判断产品口碑的正负向，回复请用一个词语：正向 或者 负向'},
    {'role': 'user', 'content': review}
]

if isStream:
    response = client.chat.completions.create(
        model='inclusionAI/Ring-2.6-1T', # ModelScope Model-Id, required
        messages=messages,
        stream=True
    )
    done_reasoning = False
    for chunk in response:
        if chunk.choices:
            reasoning_chunk = chunk.choices[0].delta.reasoning_content
            answer_chunk = chunk.choices[0].delta.content
            if reasoning_chunk != '':
                print(reasoning_chunk, end='', flush=True)
            elif answer_chunk != '':
                if not done_reasoning:
                    print('\n\n === Final Answer ===\n')
                    done_reasoning = True
                print(answer_chunk, end='', flush=True)

else:
    response = client.chat.completions.create(
        model='inclusionAI/Ring-2.6-1T', # ModelScope Model-Id, required
        messages=messages,
        stream=False
    )

    print(f'评论内容：{review}')
    print(f'情感分析结果：{response.choices[0].message.content}')
