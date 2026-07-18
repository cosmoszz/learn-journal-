import accelerate
import os
import glob

#pip install llama-cpp-python --no-cache-dir --force-reinstall --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu126 -i http://mirrors.aliyun.com/pypi/simple/
def load_model(model_path, device_map="auto",
               torch_dtype="auto", trust_remote_code=True):
    """
    使用 Transformers 加载标准 PyTorch 格式模型（适用于 HuggingFace 原版格式）
    注意：不能用于 GGUF 格式模型
    """
    print("\n正在测试加载模型（Transformers）...")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # 配置 4-bit 量化
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
    )
    print(f"[OK] 模型加载成功！参数量: {model.num_parameters() / 1e9:.2f}B")
    return model, tokenizer


def load_gguf_model(
    model_dir,
    gguf_filename=None,
    n_ctx=2048,
    n_gpu_layers=0,
    verbose=False,
):
    """
    使用 llama-cpp-python 加载 GGUF 格式模型

    参数:
        model_dir: GGUF 模型所在目录（包含 .gguf 文件）
        gguf_filename: 指定要加载的 .gguf 文件名（如 "gemma-4-E4B-it-Q4_K_M.gguf"）
                       若为 None，则自动选择目录中最大的非多模态 .gguf 文件
        n_ctx: 上下文窗口大小
        n_gpu_layers: 卸载到 GPU 的层数（0=纯CPU，建议 20+ 可加速）
        verbose: 是否显示详细日志
    """
    print("\n正在加载 GGUF 模型（llama-cpp-python）...")

    if gguf_filename is None:
        # 自动选择最佳 GGUF 文件：
        # 1. 排除 mmproj 多模态投影文件
        # 2. 选择 Q4_K_M 量化（质量/速度平衡），如果没有则选最大的
        gguf_files = [f for f in os.listdir(model_dir)
                      if f.endswith(".gguf") and not f.startswith("mmproj")]

        if not gguf_files:
            raise FileNotFoundError(f"在 {model_dir} 下未找到 .gguf 文件")

        # 优先选择 Q4_K_M 量化
        preferred = [f for f in gguf_files if "Q4_K_M" in f]
        if preferred:
            gguf_filename = preferred[0]
        else:
            # 否则选文件最大的（通常质量最高量化最轻）
            gguf_files.sort(key=lambda f: os.path.getsize(os.path.join(model_dir, f)), reverse=True)
            gguf_filename = gguf_files[0]

    gguf_path = os.path.join(model_dir, gguf_filename)

    if not os.path.exists(gguf_path):
        raise FileNotFoundError(f"GGUF 文件不存在: {gguf_path}")

    file_size_gb = os.path.getsize(gguf_path) / (1024 ** 3)
    print(f"  [GGUF 文件]: {gguf_filename}")
    print(f"  [文件大小]: {file_size_gb:.2f} GB")
    print(f"  [上下文窗口]: {n_ctx}")
    print(f"  [GPU 层数]: {n_gpu_layers} ({'纯 CPU' if n_gpu_layers == 0 else f'{n_gpu_layers} 层卸载到 GPU'})")

    from llama_cpp import Llama

    llm = Llama(
        model_path=gguf_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        verbose=verbose,
    )

    print(f"[OK] GGUF 模型加载成功！")
    return llm


def load_gguf_model_with_info(
    gguf_filepath,
    n_ctx=2048,
    n_gpu_layers=0,
    verbose=False,
    test_inference=True,
    test_message="你好，请用一句话介绍你自己",
    test_max_tokens=100,
    test_temperature=0.7,
):
    """
    使用完整 GGUF 文件路径加载模型，可选进行推理测试。

    参数:
        gguf_filepath: GGUF 文件的完整路径（如 "D:/models/gemma-4-E4B-it-Q4_K_M.gguf"）
        n_ctx: 上下文窗口大小
        n_gpu_layers: 卸载到 GPU 的层数（-1=全部卸到GPU, 0=纯CPU）
        verbose: 是否显示 llama-cpp 详细日志
        test_inference: 是否进行推理测试
        test_message: 测试用的用户消息
        test_max_tokens: 测试最大 token 数
        test_temperature: 测试温度参数

    返回:
        llm: Llama 模型对象
    """
    print("\n正在加载 GGUF 模型（llama-cpp-python）...")

    if not os.path.exists(gguf_filepath):
        raise FileNotFoundError(f"GGUF 文件不存在: {gguf_filepath}")

    file_size_gb = os.path.getsize(gguf_filepath) / (1024 ** 3)
    print(f"  [GGUF 文件]: {os.path.basename(gguf_filepath)}")
    print(f"  [文件大小]: {file_size_gb:.2f} GB")
    print(f"  [上下文窗口]: {n_ctx}")
    print(f"  [GPU 层数]: {n_gpu_layers} ({'纯 CPU' if n_gpu_layers == 0 else f'{n_gpu_layers} 层卸载到 GPU'})")

    from llama_cpp import Llama

    llm = Llama(
        model_path=gguf_filepath,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        verbose=verbose,
    )

    print(f"[OK] GGUF 模型加载成功！")

    # ---------- 可选：测试推理 ----------
    if test_inference:
        print("\n[TEST] 测试推理...")
        try:
            response = llm.create_chat_completion(
                messages=[{"role": "user", "content": test_message}],
                max_tokens=test_max_tokens,
                temperature=test_temperature,
            )
            test_response = response['choices'][0]['message']['content']
            print(f"  模型回答:\n{test_response}")
        except Exception as e:
            print(f"  [WARN] 推理测试失败: {e}")

    print("\n[OK] 模型加载完成！可通过返回的 llm 对象进行推理。")
    return llm


if __name__ == "__main__":
    import sys
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    GGUF_PATH = os.path.join(project_root, "model_cache/model_save/unsloth/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf")

    # 加载模型（只返回 llm）
    llm = load_gguf_model_with_info(
        gguf_filepath=GGUF_PATH,
        n_ctx=2048,
        n_gpu_layers=-1,
        verbose=False,
        test_inference=True,
        test_message="你好，请用一句话介绍你自己",
        test_max_tokens=100,
        test_temperature=0.7,
    )

    print(f"\nllm 对象类型: {type(llm)}")
    print("模型已加载到内存，可继续推理使用。")
    llm.close()
