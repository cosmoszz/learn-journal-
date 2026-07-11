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
    model_dir,
    gguf_filename=None,
    n_ctx=2048,
    n_gpu_layers=0,
    verbose=False,
    test_inference=True,
    test_message="你好，请用一句话介绍你自己",
    test_max_tokens=100,
    test_temperature=0.7,
    auto_close=False,
):
    """
    加载 GGUF 模型并获取模型信息（层数等），可选进行推理测试。

    这是 load_gguf_model 的增强版，封装了加载后的信息获取与测试流程。

    参数:
        model_dir: GGUF 模型所在目录
        gguf_filename: GGUF 文件名，若为 None 则自动选择
        n_ctx: 上下文窗口大小
        n_gpu_layers: 卸载到 GPU 的层数（-1=全部卸到GPU, 0=纯CPU）
        verbose: 是否显示 llama-cpp 详细日志
        test_inference: 是否进行推理测试
        test_message: 测试用的用户消息
        test_max_tokens: 测试最大 token 数
        test_temperature: 测试温度参数
        auto_close: 测试完毕后是否自动关闭模型（关闭后无法再推理）

    返回:
        (llm, info) 元组
        - llm: Llama 模型对象（若 auto_close=True 则为 None）
        - info: dict，包含模型信息
            {
                "gguf_filename": 加载的 GGUF 文件名,
                "gguf_path": 完整路径,
                "file_size_gb": 文件大小(GB),
                "n_ctx": 上下文大小,
                "n_gpu_layers": GPU 层数,
                "layer_count": 模型总层数(int, 若无法获取则为 None),
                "metadata": 模型原始 metadata 字典,
                "test_response": 测试推理结果(若 test_inference=True),
            }
    """
    # ---------- 1. 加载模型 ----------
    llm = load_gguf_model(
        model_dir=model_dir,
        gguf_filename=gguf_filename,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        verbose=verbose,
    )

    # ---------- 2. 提取模型信息 ----------
    print("\n[INFO] 提取模型信息...")

    # 确定实际加载的文件名
    # load_gguf_model 内部可能自动选择了文件，但从外部无法直接获取
    # 我们通过 gguf_path 来反推
    if gguf_filename is None:
        # 从 model_dir 中再次匹配（与 load_gguf_model 逻辑一致）
        gguf_files = [f for f in os.listdir(model_dir)
                      if f.endswith(".gguf") and not f.startswith("mmproj")]
        preferred = [f for f in gguf_files if "Q4_K_M" in f]
        if preferred:
            actual_filename = preferred[0]
        else:
            gguf_files.sort(key=lambda f: os.path.getsize(os.path.join(model_dir, f)), reverse=True)
            actual_filename = gguf_files[0]
    else:
        actual_filename = gguf_filename

    gguf_path = os.path.join(model_dir, actual_filename)
    file_size_gb = os.path.getsize(gguf_path) / (1024 ** 3)

    # 获取模型 metadata
    metadata = {}
    if hasattr(llm, "metadata") and isinstance(llm.metadata, dict):
        metadata = llm.metadata

    # 自动检测架构名并获取层数
    layer_count = None
    if metadata:
        # 常见架构名映射到对应的 block_count key
        # 格式: {architecture_name: block_count_key}
        arch_block_keys = {
            "gemma4": "gemma4.block_count",
            "gemma2": "gemma2.block_count",
            "llama": "llama.block_count",
            "mistral": "mistral.block_count",
            "mixtral": "mixtral.block_count",
            "qwen2": "qwen2.block_count",
            "qwen2_moe": "qwen2_moe.block_count",
            "deepseek2": "deepseek2.block_count",
            "phi3": "phi3.block_count",
            "stablelm": "stablelm.block_count",
            "starcoder2": "starcoder2.block_count",
            "falcon": "falcon.block_count",
            "command_r": "command_r.block_count",
            "dbrx": "dbrx.block_count",
            "bert": "bert.block_count",
            "nomic": "nomic.block_count",
            "olmo": "olmo.block_count",
            "openelm": "openelm.block_count",
            "nemotron": "nemotron.block_count",
            "exaone": "exaone.block_count",
        }

        # 先尝试通过 general.architecture 获取架构名
        arch = metadata.get("general.architecture")
        if arch and arch in arch_block_keys:
            key = arch_block_keys[arch]
            val = metadata.get(key)
            if val is not None:
                try:
                    layer_count = int(val)
                except (ValueError, TypeError):
                    pass

        # 如果没找到，遍历所有可能的 block_count key
        if layer_count is None:
            for key in arch_block_keys.values():
                val = metadata.get(key)
                if val is not None:
                    try:
                        layer_count = int(val)
                        break
                    except (ValueError, TypeError):
                        continue

        # 最后兜底：扫描 metadata 中所有包含 "block_count" 的 key
        if layer_count is None:
            for k, v in metadata.items():
                if "block_count" in k:
                    try:
                        layer_count = int(v)
                        print(f"  [自动检测] 从 {k} 获取到层数: {layer_count}")
                        break
                    except (ValueError, TypeError):
                        continue

    if layer_count is not None:
        print(f"  模型总层数: {layer_count}")
    else:
        print("  未能获取模型层数（metadata 中无 block_count 信息）")

    # ---------- 3. 可选：测试推理 ----------
    test_response = None
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
            test_response = None

    # ---------- 4. 组装 info 字典 ----------
    info = {
        "gguf_filename": actual_filename,
        "gguf_path": gguf_path,
        "file_size_gb": round(file_size_gb, 2),
        "n_ctx": n_ctx,
        "n_gpu_layers": n_gpu_layers,
        "layer_count": layer_count,
        "metadata": metadata,
        "test_response": test_response,
    }

    # ---------- 5. 可选：自动关闭 ----------
    if auto_close:
        print("\n[INFO] auto_close=True，关闭模型...")
        llm.close()
        print("  模型已关闭")
        return None, info

    print("\n[OK] 模型加载完成！可通过返回的 llm 对象继续推理，或通过 info 查看模型信息。")
    return llm, info


if __name__ == "__main__":
    import sys
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    MODEL_DIR = os.path.join(project_root, "model_cache/model_save/unsloth/gemma-4-E4B-it-GGUF")

    # 使用封装后的函数
    llm, info = load_gguf_model_with_info(
        model_dir=MODEL_DIR,
        gguf_filename="gemma-4-E4B-it-Q4_K_M.gguf",
        n_ctx=2048,
        n_gpu_layers=-1,  # -1 = 全部卸载到 GPU
        verbose=False,
        test_inference=True,
        test_message="你好，请用一句话介绍你自己",
        test_max_tokens=100,
        test_temperature=0.7,
        auto_close=True,  # 测试完毕后自动关闭
    )

    # 打印汇总信息
    print("\n" + "=" * 50)
    print("模型信息汇总:")
    print(f"  GGUF 文件: {info['gguf_filename']}")
    print(f"  文件大小: {info['file_size_gb']} GB")
    print(f"  模型层数: {info['layer_count']}")
    print(f"  上下文窗口: {info['n_ctx']}")
    print(f"  GPU 层数: {info['n_gpu_layers']}")
    if info['test_response']:
        print(f"  测试回答: {info['test_response'][:50]}...")
    print("=" * 50)