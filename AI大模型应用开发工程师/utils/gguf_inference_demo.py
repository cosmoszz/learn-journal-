"""
GGUF 模型推理测试 Demo

演示如何使用 load_gguf_model_with_info 加载模型并进行多轮对话。
"""

from utils.load_model_from_cache import load_gguf_model_with_info
import os

# ---------- 1. 配置路径 ----------
# 根据实际模型路径修改
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
GGUF_PATH = os.path.join(project_root, "model_cache/model_save/unsloth/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf")

# ---------- 2. 加载模型（自动进行一轮推理测试） ----------
llm = load_gguf_model_with_info(
    gguf_filepath=GGUF_PATH,  # GGUF 文件完整路径
    n_ctx=2048,               # 上下文窗口大小
    n_gpu_layers=-1,          # -1 = 全部卸载到 GPU
    verbose=False,            # 不显示 llama-cpp 详细日志
    test_inference=True,      # 加载完成后自动进行一次推理测试
    test_message="你好，请用一句话介绍你自己",
    test_max_tokens=100000,
    test_temperature=0.7,
)

print("\n" + "=" * 60)

# ---------- 3. 多轮对话 Demo ----------
messages = [
    {"role": "system", "content": "你是一个有用的AI助手。"},
    {"role": "user", "content": "请用三句话解释什么是大语言模型（LLM）。"},
]

print("\n[用户]:", messages[-1]["content"])

response = llm.create_chat_completion(
    messages=messages,
    max_tokens=200,
    temperature=0.5,
)

assistant_reply = response['choices'][0]['message']['content']
print("[助手]:", assistant_reply)

# ---------- 4. 继续对话 ----------
messages.append({"role": "assistant", "content": assistant_reply})
messages.append({"role": "user", "content": "刚才说的这些，和传统的机器学习模型有什么主要区别？"})

print("\n[用户]:", messages[-1]["content"])

response = llm.create_chat_completion(
    messages=messages,
    max_tokens=200,
    temperature=0.7,
)

assistant_reply = response['choices'][0]['message']['content']
print("[助手]:", assistant_reply)

print("\n" + "=" * 60)
print("Demo 运行完毕！")

# ---------- 5. 使用完毕后关闭模型释放资源 ----------
llm.close()
print("模型已关闭，资源已释放。")