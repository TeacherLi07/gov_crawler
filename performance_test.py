import asyncio
import time
import statistics
from openai import AsyncOpenAI
import concurrent.futures
from datetime import datetime
from typing import List, Tuple
import json

# 初始化异步 OpenAI 客户端
client = AsyncOpenAI(
    api_key="not-needed",
    base_url="http://localhost:8000/v1"
)

# 测试样例：生成 1000 tokens 的输入
TEST_PROMPTS = [
    """请详细分析以下问题并提供解决方案。

问题背景：在现代企业中，数据处理和分析变得越来越重要。许多组织都在尝试利用人工智能和机器学习技术来优化他们的业务流程。

具体问题描述：
1. 如何在大规模数据集上有效地应用机器学习算法？
2. 如何确保数据质量和隐私保护？
3. 如何评估机器学习模型的性能？
4. 如何将模型部署到生产环境中？
5. 如何处理模型更新和维护？

请从以下几个方面进行详细分析：
- 数据采集和预处理阶段的最佳实践
- 特征工程和选择的方法论
- 模型选择和超参数调优的策略
- 交叉验证和测试集评估的重要性
- 部署前的准备工作和注意事项

此外，还需要考虑：
- 成本效益分析
- 团队技能要求
- 时间投入预估
- 风险管理措施
- 长期维护计划

请提供一个完整的解决方案框架，包括具体的步骤、工具选择、资源需求和预期结果。""",
    
    """分析以下技术架构的优缺点。

背景信息：我们正在考虑将现有的单体应用迁移到微服务架构。当前系统处理每天数百万级别的请求，用户分布在全球。

现有架构的特点：
- 单一代码库
- 共享数据库
- 部署时需要停服
- 扩展困难
- 技术栈统一

提议的微服务架构包括：
- API 网关
- 服务注册与发现
- 容器化部署
- 消息队列通信
- 分布式数据库

请分析：
1. 微服务架构相比单体的优势
2. 迁移过程中可能遇到的困难
3. 所需的基础设施投入
4. 团队能力要求的变化
5. 潜在的性能影响
6. 安全性考虑
7. 成本估算

还要评估：
- 迁移的复杂度
- 实施的时间表
- 回滚方案
- 监控和告警体系
- 灾难恢复计划""",
    
    """探讨人工智能在医疗诊断中的应用。

当前医疗现状：
- 医生工作负担重
- 诊断错误率存在
- 患者等待时间长
- 医疗资源分配不均

AI 应用潜力：
- 影像识别
- 病情预测
- 个性化治疗
- 药物发现

具体应用场景分析：
1. 医学影像诊断（CT、MRI、X光）
2. 病理诊断和分类
3. 预测患者预后
4. 识别高风险患者
5. 辅助手术规划

需要考虑的因素：
- 算法准确性和可靠性
- 医疗法规和合规性
- 患者隐私和数据安全
- 医生的接受度
- 实施成本
- 培训需求
- 伦理考虑
- 责任划分

请评估：
- 技术可行性
- 临床有效性
- 经济可行性
- 实施计划
- 风险管理
- 监管审批路径"""
]

class PerformanceTester:
    def __init__(self, model: str = "models/qwen3-14b-awq"):
        self.model = model
        self.results = []
        self.start_time = None
        self.end_time = None
    
    async def send_request(self, request_id: int, prompt: str) -> Tuple[int, float, float, float]:
        """
        发送单个请求并记录性能指标
        返回: (request_id, response_time, input_tokens, output_tokens)
        """
        try:
            start = time.time()
            
            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的技术顾问和分析师。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                frequency_penalty=0.2,
                max_tokens=256,
                timeout=300  # 5分钟超时
            )
            
            end = time.time()
            response_time = end - start
            
            # 获取 token 计数
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
            
            return (request_id, response_time, input_tokens, output_tokens)
        
        except Exception as e:
            print(f"请求 {request_id} 失败: {e}")
            return (request_id, None, 0, 0)
    
    async def run_concurrent_test(self, concurrency: int, total_requests: int) -> dict:
        """
        运行并发测试
        """
        print(f"\n开始测试并发数: {concurrency}, 总请求数: {total_requests}")
        
        self.start_time = time.time()
        tasks = []
        
        # 创建所有任务
        for i in range(total_requests):
            prompt = TEST_PROMPTS[i % len(TEST_PROMPTS)]
            task = self.send_request(i, prompt)
            tasks.append(task)
        
        # 限制并发数量
        semaphore = asyncio.Semaphore(concurrency)
        
        async def bounded_request(task):
            async with semaphore:
                return await task
        
        # 执行所有任务
        results = await asyncio.gather(*[bounded_request(task) for task in tasks])
        
        self.end_time = time.time()
        total_time = self.end_time - self.start_time
        
        # 过滤掉失败的请求
        successful_results = [r for r in results if r[1] is not None]
        
        if not successful_results:
            print("所有请求都失败了!")
            return None
        
        # 计算统计信息
        response_times = [r[1] for r in successful_results]
        input_tokens_list = [r[2] for r in successful_results]
        output_tokens_list = [r[3] for r in successful_results]
        
        total_input_tokens = sum(input_tokens_list)
        total_output_tokens = sum(output_tokens_list)
        successful_requests = len(successful_results)
        failed_requests = total_requests - successful_requests
        
        # 计算吞吐量
        total_tokens = total_input_tokens + total_output_tokens
        tokens_per_second = total_tokens / total_time if total_time > 0 else 0
        requests_per_second = successful_requests / total_time if total_time > 0 else 0
        
        stats = {
            "concurrency": concurrency,
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "total_time": total_time,
            "avg_response_time": statistics.mean(response_times),
            "min_response_time": min(response_times),
            "max_response_time": max(response_times),
            "median_response_time": statistics.median(response_times),
            "p95_response_time": statistics.quantiles(response_times, n=20)[18] if len(response_times) > 20 else max(response_times),
            "p99_response_time": statistics.quantiles(response_times, n=100)[98] if len(response_times) > 100 else max(response_times),
            "stdev_response_time": statistics.stdev(response_times) if len(response_times) > 1 else 0,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_tokens,
            "tokens_per_second": tokens_per_second,
            "requests_per_second": requests_per_second,
            "avg_input_tokens": statistics.mean(input_tokens_list),
            "avg_output_tokens": statistics.mean(output_tokens_list),
        }
        
        return stats
    
    def print_results(self, stats: dict):
        """打印测试结果"""
        if stats is None:
            return
        
        print("\n" + "=" * 80)
        print(f"并发数: {stats['concurrency']} | 总请求: {stats['total_requests']}")
        print("=" * 80)
        print(f"成功请求: {stats['successful_requests']} | 失败请求: {stats['failed_requests']}")
        print(f"总耗时: {stats['total_time']:.2f}s")
        print()
        print("响应时间统计 (秒):")
        print(f"  平均: {stats['avg_response_time']:.2f}s")
        print(f"  最小: {stats['min_response_time']:.2f}s")
        print(f"  最大: {stats['max_response_time']:.2f}s")
        print(f"  中位数: {stats['median_response_time']:.2f}s")
        print(f"  P95: {stats['p95_response_time']:.2f}s")
        print(f"  P99: {stats['p99_response_time']:.2f}s")
        print(f"  标准差: {stats['stdev_response_time']:.2f}s")
        print()
        print("Token 统计:")
        print(f"  平均输入 tokens: {stats['avg_input_tokens']:.0f}")
        print(f"  平均输出 tokens: {stats['avg_output_tokens']:.0f}")
        print(f"  总输入 tokens: {stats['total_input_tokens']}")
        print(f"  总输出 tokens: {stats['total_output_tokens']}")
        print(f"  总 tokens: {stats['total_tokens']}")
        print()
        print("吞吐量:")
        print(f"  Tokens/秒: {stats['tokens_per_second']:.2f}")
        print(f"  请求/秒: {stats['requests_per_second']:.2f}")
        print("=" * 80)


async def main():
    """主测试函数"""
    print("\n" + "=" * 80)
    print("vLLM 并发性能测试")
    print("=" * 80)
    print(f"测试参数设置:")
    print(f"  - 模型: models/qwen3-14b-awq")
    print(f"  - 输入长度: ~1000 tokens")
    print(f"  - 输出长度: 256 tokens (max_tokens)")
    print(f"  - Temperature: 0.1")
    print(f"  - Frequency Penalty: 0.2")
    print(f"  - 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    tester = PerformanceTester()
    
    # 测试不同的并发数
    concurrency_levels = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    requests_per_level = 4  # 每个并发级别测试 10 个请求
    
    all_stats = []
    
    for concurrency in concurrency_levels:
        stats = await tester.run_concurrent_test(
            concurrency=concurrency,
            total_requests=requests_per_level * concurrency
        )
        
        if stats:
            tester.print_results(stats)
            all_stats.append(stats)
        
        # 请求之间稍作延迟，避免服务器过载
        await asyncio.sleep(5)
    
    # 生成总结报告
    print("\n" + "=" * 80)
    print("性能测试总结")
    print("=" * 80)
    
    if all_stats:
        # 找到吞吐量最高的并发数
        max_tps_stat = max(all_stats, key=lambda x: x['tokens_per_second'])
        max_rps_stat = max(all_stats, key=lambda x: x['requests_per_second'])
        
        print("\n吞吐量排名 (Tokens/秒):")
        print("-" * 80)
        print(f"{'并发数':<10} {'Tokens/秒':<20} {'请求/秒':<20} {'平均响应时间':<20}")
        print("-" * 80)
        
        sorted_stats = sorted(all_stats, key=lambda x: x['tokens_per_second'], reverse=True)
        for i, stat in enumerate(sorted_stats, 1):
            print(f"{stat['concurrency']:<10} {stat['tokens_per_second']:<20.2f} "
                  f"{stat['requests_per_second']:<20.2f} {stat['avg_response_time']:<20.2f}s")
        
        print("-" * 80)
        print(f"\n最优并发数 (Tokens/秒): {max_tps_stat['concurrency']}")
        print(f"  - 吞吐量: {max_tps_stat['tokens_per_second']:.2f} tokens/秒")
        print(f"  - 请求量: {max_tps_stat['requests_per_second']:.2f} 请求/秒")
        print(f"  - 平均响应时间: {max_tps_stat['avg_response_time']:.2f}秒")
        
        # 保存详细报告
        report = {
            "timestamp": datetime.now().isoformat(),
            "test_parameters": {
                "model": "models/qwen3-14b-awq",
                "input_tokens": "~1000",
                "output_tokens": "256",
                "temperature": 0.1,
                "frequency_penalty": 0.2
            },
            "results": all_stats
        }
        
        with open("performance_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print("\n详细报告已保存到 performance_report.json")
    
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
