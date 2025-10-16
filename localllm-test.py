from openai import OpenAI
import json

# 初始化 OpenAI 客户端，指向本地 vLLM 服务
client = OpenAI(
    api_key="not-needed",  # vLLM 不需要真实的 API 密钥
    base_url="http://localhost:8000/v1"
)

def test_basic_completion():
    """基础测试：简单的文本生成"""
    print("=" * 60)
    print("测试 1: 基础完成 (Basic Completion)")
    print("=" * 60)
    
    response = client.completions.create(
        model="/models/qwen3-8b",
        prompt="请介绍一下中国的首都北京",
        max_tokens=100
    )
    
    print(f"Prompt: 请介绍一下中国的首都北京")
    print(f"Response: {response.choices[0].text}\n")


def test_chat_completion():
    """聊天模式测试"""
    print("=" * 60)
    print("测试 2: 聊天完成 (Chat Completion)")
    print("=" * 60)
    
    response = client.chat.completions.create(
        model="/models/qwen3-8b",
        messages=[
            {"role": "system", "content": "你是一个有帮助的AI助手。"},
            {"role": "user", "content": "用5句话总结Python的主要特点"}
        ],
        max_tokens=150
    )
    
    print(f"System: 你是一个有帮助的AI助手。")
    print(f"User: 用5句话总结Python的主要特点")
    print(f"Assistant: {response.choices[0].message.content}\n")


def test_temperature_variations():
    """温度参数测试：控制生成的随机性"""
    print("=" * 60)
    print("测试 3: 温度参数变化 (Temperature Variations)")
    print("=" * 60)
    print("温度越低，生成越确定；温度越高，生成越随机")
    print()
    
    temperatures = [0.1, 0.7, 1.5]
    prompt = "完成这个故事：从前有一个"
    
    for temp in temperatures:
        response = client.completions.create(
            model="/models/qwen3-8b",
            prompt=prompt,
            temperature=temp,
            max_tokens=50
        )
        print(f"Temperature={temp}: {response.choices[0].text}")
        print()


def test_top_p_sampling():
    """Top-P 采样测试：核采样"""
    print("=" * 60)
    print("测试 4: Top-P 采样 (Nucleus Sampling)")
    print("=" * 60)
    print("Top-P 控制生成的多样性，范围 0-1")
    print()
    
    top_p_values = [0.5, 0.9, 0.95]
    prompt = "人工智能的未来发展方向是"
    
    for top_p in top_p_values:
        response = client.completions.create(
            model="/models/qwen3-8b",
            prompt=prompt,
            top_p=top_p,
            temperature=0.8,
            max_tokens=60
        )
        print(f"Top-P={top_p}: {response.choices[0].text}")
        print()


def test_max_tokens():
    """最大令牌数测试"""
    print("=" * 60)
    print("测试 5: 最大令牌数 (Max Tokens)")
    print("=" * 60)
    
    max_tokens_list = [30, 80, 150]
    prompt = "请详细解释什么是机器学习"
    
    for max_tok in max_tokens_list:
        response = client.completions.create(
            model="/models/qwen3-8b",
            prompt=prompt,
            max_tokens=max_tok
        )
        text = response.choices[0].text
        print(f"Max Tokens={max_tok}:")
        print(f"  生成长度: {len(text)}")
        print(f"  内容: {text[:100]}{'...' if len(text) > 100 else ''}")
        print()


def test_frequency_penalty():
    """频率惩罚测试：减少重复"""
    print("=" * 60)
    print("测试 6: 频率惩罚 (Frequency Penalty)")
    print("=" * 60)
    print("频率惩罚越高，越不容易重复词语")
    print()
    
    penalties = [0.0, 0.5, 1.5]
    prompt = "列举编程语言的优点："
    
    for penalty in penalties:
        response = client.completions.create(
            model="/models/qwen3-8b",
            prompt=prompt,
            frequency_penalty=penalty,
            max_tokens=80
        )
        print(f"Frequency Penalty={penalty}: {response.choices[0].text}")
        print()


def test_presence_penalty():
    """存在惩罚测试：鼓励新话题"""
    print("=" * 60)
    print("测试 7: 存在惩罚 (Presence Penalty)")
    print("=" * 60)
    print("存在惩罚越高，越容易引入新的概念")
    print()
    
    penalties = [0.0, 0.8, 1.5]
    prompt = "介绍几个不同的文化"
    
    for penalty in penalties:
        response = client.completions.create(
            model="/models/qwen3-8b",
            prompt=prompt,
            presence_penalty=penalty,
            max_tokens=100
        )
        print(f"Presence Penalty={penalty}: {response.choices[0].text}")
        print()


def test_combined_parameters():
    """组合参数测试：最优化设置"""
    print("=" * 60)
    print("测试 8: 组合参数优化")
    print("=" * 60)
    
    response = client.chat.completions.create(
        model="/models/qwen3-8b",
        messages=[
            {"role": "system", "content": "你是一个专业的技术写手"},
            {"role": "user", "content": "写一个关于云计算的简短介绍"}
        ],
        temperature=0.7,          # 平衡的随机性
        top_p=0.9,               # 高质量采样
        max_tokens=200,          # 足够长的输出
        frequency_penalty=0.6,   # 减少重复
        presence_penalty=0.3,    # 适度引入新概念
    )
    
    print("参数设置:")
    print(f"  temperature: 0.7")
    print(f"  top_p: 0.9")
    print(f"  max_tokens: 200")
    print(f"  frequency_penalty: 0.6")
    print(f"  presence_penalty: 0.3")
    print()
    print(f"生成结果:\n{response.choices[0].message.content}\n")


def test_model_info():
    """获取模型信息"""
    print("=" * 60)
    print("测试 9: 获取可用模型")
    print("=" * 60)
    
    models = client.models.list()
    print("可用模型:")
    for model in models.data:
        print(f"  - {model.id}")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("vLLM Qwen3-8B-AWQ 测试套件")
    print("=" * 60 + "\n")
    
    try:
        # 按顺序运行测试
        test_model_info()
        test_basic_completion()
        test_chat_completion()
        test_temperature_variations()
        test_top_p_sampling()
        test_max_tokens()
        test_frequency_penalty()
        test_presence_penalty()
        test_combined_parameters()
        
        print("=" * 60)
        print("所有测试完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"错误: {e}")
        print("请确保 vLLM 服务正在运行在 localhost:8000")
