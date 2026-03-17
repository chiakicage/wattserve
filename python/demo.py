import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 设置本地模型路径
model_path = "/share/models/Qwen3-4B"

print("正在加载模型，请稍候...")

# 2. 加载 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 3. 加载模型
# device_map="auto" 会自动将模型加载到可用的 GPU 上
# torch_dtype="auto" 会自动使用模型原本的精度（通常是 bf16 或 fp16）
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype="auto", device_map="auto"
)

print("模型加载完成！输入 'exit' 或 'quit' 可以退出对话。\n")
print("-" * 50)

# 4. 进入交互式对话循环
while True:
    # 获取用户输入
    # user_input = input("你: ")
    user_input = "Please proof the Fermat's Last Theorem in detail."

    # 退出条件
    if user_input.strip().lower() in ["exit", "quit"]:
        print("对话结束。")
        break

    if not user_input.strip():
        continue

    # 5. 构建符合 Qwen 格式的消息模板
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_input},
    ]

    # 将消息列表转换为模型期望的 prompt 字符串
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # 将文本转换为 token 并放入模型所在的设备 (如 GPU)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # 6. 生成回复
    torch.cuda.synchronize()  # 确保前面的输入处理完成
    start_time = time.time()
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=8192,  # 控制最大生成长度
            temperature=0,  # 控制生成的随机性，值越小越确定
            do_sample=False,  # 启用采样
        )
    torch.cuda.synchronize()  # 确保生成完成
    end_time = time.time()

    # 7. 截取并解码生成的这部分 token（剔除掉输入 prompt 的部分）
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    output_len = sum(len(ids) for ids in generated_ids)
    token_per_sec = output_len / (end_time - start_time)

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
        0
    ]

    # 打印模型输出
    print(f"Qwen: {response}\n")
    print(f"Performance: {token_per_sec:.2f}/s")
    break
