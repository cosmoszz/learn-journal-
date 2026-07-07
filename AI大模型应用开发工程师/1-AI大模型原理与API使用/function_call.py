import sys
import io
import os
import json

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url=os.getenv('MODELSCOPE_BASE_URL'),
    api_key=os.getenv('MODELSCOPE_API_KEY'),
)

# 定义天气查询工具
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，例如：北京、上海、深圳"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

# 模拟天气查询函数
def get_weather(city):
    weather_data = {
        "北京": {"temperature": "25°C", "weather": "晴", "humidity": "45%"},
        "上海": {"temperature": "28°C", "weather": "多云", "humidity": "65%"},
        "深圳": {"temperature": "32°C", "weather": "阵雨", "humidity": "80%"},
        "广州": {"temperature": "30°C", "weather": "雷阵雨", "humidity": "75%"},
        "杭州": {"temperature": "26°C", "weather": "阴", "humidity": "60%"},
    }
    return weather_data.get(city, {"temperature": "未知", "weather": "未知", "humidity": "未知"})

def run_conversation():
    messages = [
        {"role": "system", "content": "你是一个有用的助手，可以帮用户查询天气。当用户询问天气时，请调用 get_weather 函数获取数据后回复。"}
    ]

    print("=== 天气查询助手 ===")
    print("输入 'quit' 或 'exit' 退出\n")

    while True:
        user_input = input("你: ")
        if user_input.lower() in ('quit', 'exit', 'q'):
            print("再见！")
            break

        messages.append({"role": "user", "content": user_input})

        # 第一次调用：让模型决定是否需要调用函数
        response = client.chat.completions.create(
            model='inclusionAI/Ring-2.6-1T',
            messages=messages,
            tools=tools,
            tool_choice="auto",
            stream=False
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        # 检查模型是否调用了工具
        if tool_calls:
            messages.append(response_message)

            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                if function_name == "get_weather":
                    city = function_args.get("city")
                    result = get_weather(city)
                    print(f"[调用工具] 查询 {city} 的天气...")

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result, ensure_ascii=False)
                    })

            # 第二次调用：让模型根据函数返回结果生成最终回复
            response = client.chat.completions.create(
                model='inclusionAI/Ring-2.6-1T',
                messages=messages,
                stream=False
            )

            assistant_message = response.choices[0].message
            print(f"助手: {assistant_message.content}\n")
            messages.append({"role": "assistant", "content": assistant_message.content})
        else:
            print(f"助手: {response_message.content}\n")
            messages.append({"role": "assistant", "content": response_message.content})

if __name__ == "__main__":
    run_conversation()
