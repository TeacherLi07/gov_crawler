import asyncio
import csv
import json
import os
import re
import time
import platform
import socket
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from urllib.parse import urljoin, urlparse
import pandas as pd
from playwright.async_api import async_playwright, Page, Browser, BrowserContext, Request, Response
import aiohttp
import logging
from dataclasses import dataclass
from openai import AsyncOpenAI
from bs4 import BeautifulSoup
import shutil

USE_LOCAL_VLLM = True

# 自定义 DNS 缓存映射
DNS_OVERRIDE = {
    'zfwzgl.www.gov.cn': '36.112.20.164',
    # 可以添加更多需要覆盖的域名
}

# 保存原始的 getaddrinfo 函数
_original_getaddrinfo = socket.getaddrinfo

def custom_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    """自定义 DNS 解析函数，优先使用本地缓存"""
    # 检查是否在覆盖列表中
    if host in DNS_OVERRIDE:
        override_ip = DNS_OVERRIDE[host]
        logger.info(f"DNS覆盖: {host} -> {override_ip}")
        # 返回自定义的 IP 地址
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, '', (override_ip, port))]
    
    # 否则使用原始的解析函数
    return _original_getaddrinfo(host, port, family, type, proto, flags)

# 应用 DNS 覆盖
socket.getaddrinfo = custom_getaddrinfo

DEBUG = 0

logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crawler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 创建网络请求专用日志记录器
network_logger = logging.getLogger('network')
network_logger.setLevel(logging.DEBUG)
network_handler = logging.FileHandler('network.log', encoding='utf-8')
network_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
network_logger.addHandler(network_handler)
network_logger.propagate = False  # 防止重复记录到主日志

def debug_log(message: str):
    if DEBUG:
        logger.debug(message)


def disable_system_proxies():
    proxy_vars = (
        "http_proxy", "https_proxy", "ftp_proxy", "all_proxy",
        "HTTP_PROXY", "HTTPS_PROXY", "FTP_PROXY", "ALL_PROXY"
    )
    cleared = [var for var in proxy_vars if os.environ.pop(var, None)]
    if cleared:
        logger.info(f"已清理系统代理环境变量: {', '.join(cleared)}")
    
    # 输出 DNS 覆盖信息
    if DNS_OVERRIDE:
        logger.info(f"已启用 DNS 本地缓存覆盖，共 {len(DNS_OVERRIDE)} 条规则:")
        for domain, ip in DNS_OVERRIDE.items():
            logger.info(f"  {domain} -> {ip}")

@dataclass
class SearchResult:
    keyword: str
    title: str
    url: str
    summary: str
    date: str
    content: str = ""
    column: str = ""
    error: str = ""


@dataclass
class CrawlProgress:
    """爬取进度记录"""
    source_city: str
    target_city: str
    completed: bool = False
    last_update: str = ""


class LLMApiClient:
    """LLM API客户端，使用OpenAI SDK调用SiliconFlow服务"""

    # 可用模型列表，按优先级排序
    AVAILABLE_MODELS = ["/models/qwen3-8b-awq"] if USE_LOCAL_VLLM else [
        "Qwen/Qwen3-8B",
        "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        # "THUDM/GLM-Z1-9B-0414",
        # "THUDM/GLM-4-9B-0414",
        "Qwen/Qwen2.5-7B-Instruct",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    ]

    def __init__(self, api_key: str, base_url: str = "http://localhost:8000/v1" if USE_LOCAL_VLLM else "https://api.siliconflow.cn/"):
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.current_model_index = 0  # 当前使用的模型索引
        self.model_usage_count = {model: 0 for model in self.AVAILABLE_MODELS}  # 记录每个模型的使用次数
        
        # 模型切换控制
        self.model_switch_lock = asyncio.Lock()  # 模型切换锁
        self.last_429_model = None  # 记录最近触发429的模型
        self.model_429_count = {model: 0 for model in self.AVAILABLE_MODELS}  # 每个模型的429错误次数

    async def __aenter__(self):
        debug_log("LLMApiClient进入上下文管理器")
        return self

    async def __aexit__(self, exc_type, exc, tb):
        debug_log("LLMApiClient退出上下文管理器")
        # 输出模型使用统计
        logger.info("模型使用统计:")
        for model, count in self.model_usage_count.items():
            logger.info(f"  {model}: {count} 次")
        
        # 输出429错误统计
        logger.info("模型429错误统计:")
        for model, count in self.model_429_count.items():
            if count > 0:
                logger.info(f"  {model}: {count} 次")
        
        return False

    def get_current_model(self) -> str:
        """获取当前使用的模型"""
        return self.AVAILABLE_MODELS[self.current_model_index]

    def switch_to_next_model(self) -> str:
        """切换到下一个模型（非线程安全，需要配合锁使用）"""
        old_model = self.get_current_model()
        self.current_model_index = (self.current_model_index + 1) % len(self.AVAILABLE_MODELS)
        new_model = self.get_current_model()
        logger.warning(f"模型切换: {old_model} -> {new_model}")
        return new_model

    async def handle_rate_limit_error(self, current_model: str) -> str:
        """处理429速率限制错误，智能切换模型"""
        async with self.model_switch_lock:
            # 记录429错误
            self.model_429_count[current_model] += 1
            
            # 检查当前模型是否已经不是触发429的那个模型
            # 这种情况说明其他并发请求已经切换过了
            now_model = self.get_current_model()
            if now_model != current_model:
                logger.info(f"模型已被其他请求切换: {current_model} -> {now_model}，无需重复切换")
                self.last_429_model = current_model
                return now_model
            
            # 检查是否是同一个模型连续触发429
            if self.last_429_model == current_model:
                logger.info(f"模型 {current_model} 连续触发429，直接使用当前模型")
                return current_model
            
            # 执行切换
            self.last_429_model = self.switch_to_next_model()
            logger.warning(f"429错误触发模型切换: {current_model} -> {self.last_429_model}")
            
            return self.last_429_model

    def extract_main_html_content(self, html: str) -> str:
        """
        预处理HTML，去除js/css等无正文内容的部分，仅保留含中文字符的元素文本，减少tokens用量。
        """
        soup = BeautifulSoup(html, "lxml")
        # 移除无关标签
        for tag in soup(['script', 'style', 'noscript', 'link', 'iframe', 'svg', 'canvas', 'meta', 'head']):
            tag.decompose()
        # 只保留含有中文字符的文本块
        chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
        texts = []
        for elem in soup.find_all(text=True):
            text = elem.strip()
            if len(text) > 0 and chinese_pattern.search(text):
                texts.append(text)
        # 合并文本，保留段落结构
        return '\n'.join(texts)

    def extract_json_from_response(self, content: str) -> Optional[str]:
        """从响应中提取JSON内容，支持Markdown代码块格式"""
        if not content:
            return None
        
        content = content.strip()
        
        # 方法1: 查找Markdown代码块
        # 尝试查找 ```json 或 ``` 开头的代码块
        code_block_starts = []
        
        # 查找所有可能的代码块起始位置
        json_marker = '```json'
        triple_backtick = '```'
        
        # 优先查找 ```json
        pos = content.find(json_marker)
        if pos != -1:
            # 找到 ```json 后面的换行符位置
            start_pos = pos + len(json_marker)
            # 跳过可能的空白字符
            while start_pos < len(content) and content[start_pos] in ' \t\n\r':
                start_pos += 1
            code_block_starts.append(('json', start_pos))
        else:
            # 如果没有 ```json，查找普通的 ```
            pos = content.find(triple_backtick)
            if pos != -1:
                start_pos = pos + len(triple_backtick)
                # 跳过可能的空白字符
                while start_pos < len(content) and content[start_pos] in ' \t\n\r':
                    start_pos += 1
                code_block_starts.append(('plain', start_pos))
        
        # 如果找到了代码块起始标记，查找结束标记
        for block_type, start_pos in code_block_starts:
            # 从起始位置后查找结束的 ```
            end_pos = content.find(triple_backtick, start_pos)
            if end_pos != -1:
                json_str = content[start_pos:end_pos].strip()
                if json_str:
                    debug_log(f"从Markdown代码块({block_type})中提取JSON: {json_str[:100]}...")
                    return json_str
        
        # 方法2: 查找第一个 { 和最后一个 } 之间的内容
        first_brace = content.find('{')
        last_brace = content.rfind('}')
        
        if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
            json_str = content[first_brace:last_brace + 1].strip()
            debug_log(f"从括号中提取JSON: {json_str[:100]}...")
            return json_str
        
        # 方法3: 直接返回原内容（可能已经是纯JSON）
        debug_log("未检测到特殊格式，尝试直接解析原内容")
        return content

    def fix_json_format(self, json_str: str) -> str:
        """尝试修复常见的JSON格式错误"""
        if not json_str:
            return json_str
        
        original_str = json_str
        fixed = False
        
        # 移除可能的BOM和零宽字符
        json_str = json_str.replace('\ufeff', '').replace('\u200b', '')
        
        # 修复1: 智能补全缺失的闭合括号
        # 统计括号数量
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        open_brackets = json_str.count('[')
        close_brackets = json_str.count(']')
        
        # 分析括号缺失情况
        missing_close_brackets = open_brackets - close_brackets
        missing_close_braces = open_braces - close_braces
        
        if missing_close_brackets > 0 or missing_close_braces > 0:
            # 找到最后一个有效字符位置（去除尾部空白）
            json_str = json_str.rstrip()
            
            # 策略：先补全方括号，再补全大括号
            # 这是因为JSON对象通常是 {...}，数组字段在对象内部 {"field": [...]}
            
            # 补全缺失的方括号
            if missing_close_brackets > 0:
                # 检查最后一个字符
                last_char = json_str[-1] if json_str else ''
                
                # 如果最后是逗号，先移除逗号再补全括号
                if last_char == ',':
                    json_str = json_str[:-1]
                    last_char = json_str[-1] if json_str else ''
                
                # 添加缺失的方括号
                json_str += ']' * missing_close_brackets
                fixed = True
                debug_log(f"JSON修复: 添加了 {missing_close_brackets} 个缺失的闭合方括号")
            
            # 补全缺失的大括号
            if missing_close_braces > 0:
                json_str += '}' * missing_close_braces
                fixed = True
                debug_log(f"JSON修复: 添加了 {missing_close_braces} 个缺失的闭合大括号")
        
        # 修复2: 移除末尾可能的逗号（在补全括号之前就应该处理）
        # 这个逻辑已经在上面处理了，这里做二次检查
        json_str = json_str.rstrip()
        if json_str.endswith(',}'):
            json_str = json_str[:-2] + '}'
            fixed = True
            debug_log("JSON修复: 移除了对象末尾的多余逗号")
        if json_str.endswith(',]'):
            json_str = json_str[:-2] + ']'
            fixed = True
            debug_log("JSON修复: 移除了数组末尾的多余逗号")
        
        # 修复3: 处理未闭合的字符串（简单处理）
        # 检查引号配对
        quote_count = json_str.count('"')
        if quote_count % 2 != 0:
            # 找到最后一个引号的位置
            last_quote_pos = json_str.rfind('"')
            # 检查最后一个引号后面是否还有内容
            if last_quote_pos >= 0 and last_quote_pos < len(json_str) - 1:
                # 在字符串末尾添加引号
                json_str += '"'
            else:
                # 如果最后一个字符就是引号，可能是其他问题
                json_str += '"'
            fixed = True
            debug_log("JSON修复: 添加了缺失的闭合引号")
        
        # 修复4: 处理特殊情况 - 数组元素后缺少逗号或括号
        # 例如: [1, 2, 3 缺少 ]
        # 这个情况已经在修复1中处理
        
        if fixed:
            logger.info(f"JSON格式已自动修复")
            debug_log(f"修复前: {original_str[:200]}...")
            debug_log(f"修复后: {json_str[:200]}...")
            
            # 验证修复后的JSON是否平衡
            final_open_braces = json_str.count('{')
            final_close_braces = json_str.count('}')
            final_open_brackets = json_str.count('[')
            final_close_brackets = json_str.count(']')
            
            if final_open_braces != final_close_braces:
                logger.warning(f"修复后大括号仍不平衡: 开={final_open_braces}, 闭={final_close_braces}")
            if final_open_brackets != final_close_brackets:
                logger.warning(f"修复后方括号仍不平衡: 开={final_open_brackets}, 闭={final_close_brackets}")
        
        return json_str

    async def select_text_blocks(self, blocks: List[Tuple[int, str]], url: str) -> Dict[str, Any]:
        if not blocks:
            return {
                "status": "error",
                "title_indices": [],
                "content_indices": [],
                "message": "无可用文本块",
                "error": "无可用文本块"
            }

        limited_blocks = blocks[:200]
        block_lines = []
        for idx, text in limited_blocks:
            snippet = text if len(text) <= 120 else text[:117] + "..."
            block_lines.append(f"{idx}. {snippet}")

        prompt = f"""请分析以下网页文本片段，找出其中的【标题编号】和【正文编号】。

编号格式说明：
- 单个编号：`[5]`
- 如有连续同类文本块，可采用区间编号：`[[5,10]]` 表示编号5到10（含端点）
- 也可混合使用：`[[1,3], 8, [15,45]]` 

输出要求：
1. `title_indices`：标题的编号（通常只有一个编号或一个连续区间）
2. `content_indices`：正文的编号（可能包含多个编号或区间，按顺序排列）
3. `status`：仅允许 `"success"` 或 `"error"`
4. `message`：若出错，请简要说明原因；否则为空字符串

输出格式要求：
仅返回一个合法的 JSON 对象，格式如下：
{{"status":"success", "title_indices":编号, "content_indices":编号, "message":""}}
请不要输出额外说明、解释或注释。

待分析的文本片段：
{chr(10).join(block_lines)}
"""

        debug_log(f"发送到LLM的URL: {url}")
        debug_log(f"提供给LLM的文本块数量: {len(limited_blocks)}")
        debug_log(f"LLM提示词前200字符: {prompt[:200]}")

        # 最多重试次数（用于非 429 错误）
        max_retries = 10
        retry_count = 0
        # 429 错误的循环轮换次数（无限制，直到成功或遇到其他错误）
        rate_limit_cycle_count = 0
        # JSON解析失败的重试次数（独立计数）
        json_parse_retry = 0
        max_json_retries = 3  # JSON解析失败最多重试3次

        while retry_count < max_retries:
            current_model = self.get_current_model()
            
            try:
                logger.info(f"使用模型: {current_model} - URL: {url[:80]}...")
                debug_log(f"尝试 #{retry_count + 1}, 429轮换 #{rate_limit_cycle_count}, JSON重试 #{json_parse_retry}, 模型: {current_model}")
                
                response = await self.client.chat.completions.create(
                    model=current_model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1024,
                    temperature=0.2,
                    frequency_penalty=0.15,
                    stream=True,
                    timeout=30
                )

                content = ""
                async for chunk in response:
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if delta and delta.content:
                        content += delta.content
                
                # 记录成功使用
                self.model_usage_count[current_model] += 1
                logger.info(f"模型 {current_model} 成功响应 - 长度: {len(content)}")
                debug_log(f"LLM完整响应长度: {len(content)}, 使用模型: {current_model}")

                if not content.strip():
                    return {
                        "status": "error",
                        "title_indices": [],
                        "content_indices": [],
                        "message": "LLM返回空内容",
                        "error": "LLM返回空内容"
                    }

                # 使用增强的JSON提取方法
                json_str = self.extract_json_from_response(content)
                
                if not json_str:
                    logger.error(f"无法从LLM响应中提取JSON {url}, 原始内容: {content[:200]}")
                    
                    # JSON提取失败，重新提交任务
                    if json_parse_retry < max_json_retries:
                        json_parse_retry += 1
                        logger.warning(f"JSON提取失败，重新提交任务 (第{json_parse_retry}/{max_json_retries}次)")
                        await asyncio.sleep(1)
                        continue
                    
                    return {
                        "status": "error",
                        "title_indices": [],
                        "content_indices": [],
                        "message": "无法提取JSON内容",
                        "error": "无法提取JSON内容"
                    }

                try:
                    # 先尝试直接解析
                    parsed = json.loads(json_str)
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON解析失败，尝试自动修复: {e}")
                    logger.warning(f"原始JSON: {json_str[:300]}...")
                    
                    # 尝试修复JSON格式
                    fixed_json_str = self.fix_json_format(json_str)
                    
                    try:
                        # 尝试解析修复后的JSON
                        parsed = json.loads(fixed_json_str)
                        logger.info("JSON自动修复成功")
                        
                    except json.JSONDecodeError as e2:
                        logger.error(f"JSON修复后仍然解析失败: {e2}")
                        logger.error(f"修复后的JSON: {fixed_json_str[:300]}...")
                        
                        # JSON解析失败，重新提交任务
                        if json_parse_retry < max_json_retries:
                            json_parse_retry += 1
                            logger.warning(f"JSON解析失败，重新提交任务 (第{json_parse_retry}/{max_json_retries}次)")
                            logger.warning(f"原始LLM响应: {content[:500]}")
                            await asyncio.sleep(1)
                            continue
                        
                        # 达到最大重试次数，返回错误
                        return {
                            "status": "error",
                            "title_indices": [],
                            "content_indices": [],
                            "message": f"JSON解析失败: {str(e2)}",
                            "error": f"JSON解析失败: {str(e2)}"
                        }
                
                # 解析成功，处理结果
                status = (parsed.get("status") or "").lower()
                message = parsed.get("message", "")
                debug_log(f"LLM解析状态: {status}, message: {message}")
                
                return {
                    "status": status,
                    "title_indices": parsed.get("title_indices", []),
                    "content_indices": parsed.get("content_indices", []),
                    "message": message,
                    "error": "" if status == "success" else (message or "LLM返回非成功状态")
                }

            except asyncio.TimeoutError:
                error_msg = f"LLM API调用超时 (模型: {current_model})"
                logger.error(f"{error_msg} - URL: {url}")
                retry_count += 1
                if retry_count < max_retries:
                    await asyncio.sleep(2)
                    continue
                else:
                    return {
                        "status": "error",
                        "title_indices": [],
                        "content_indices": [],
                        "message": "",
                        "error": error_msg
                    }

            except Exception as e:
                error_str = str(e)
                
                # 检查是否是 429 错误（速率限制）
                if "429" in error_str or "rate_limit" in error_str.lower() or "too many requests" in error_str.lower():
                    logger.warning(f"模型 {current_model} 遇到速率限制 (429)")
                    
                    # 使用新的智能切换方法
                    new_model = await self.handle_rate_limit_error(current_model)
                    
                    # 记录轮换次数
                    rate_limit_cycle_count += 1
                    
                    # 如果已经轮换了一圈所有模型
                    if rate_limit_cycle_count > 0 and rate_limit_cycle_count % len(self.AVAILABLE_MODELS) == 0:
                        cycle_num = rate_limit_cycle_count // len(self.AVAILABLE_MODELS)
                        logger.warning(f"所有 {len(self.AVAILABLE_MODELS)} 个模型均遇到限制，第 {cycle_num} 轮轮换，等待后继续...")
                        # 等待更长时间后继续尝试
                        await asyncio.sleep(2)
                    else:
                        # 短暂延迟后重试
                        await asyncio.sleep(0.5)
                    
                    # 429 错误不计入 retry_count，持续尝试
                    continue
                else:
                    # 其他错误计入重试次数
                    error_msg = f"LLM API调用失败 (模型: {current_model}): {error_str}"
                    logger.error(f"{error_msg} - URL: {url}")
                    
                    retry_count += 1
                    if retry_count < max_retries:
                        await asyncio.sleep(2)
                        continue
                    else:
                        return {
                            "status": "error",
                            "title_indices": [],
                            "content_indices": [],
                            "message": "",
                            "error": error_msg
                        }

        # 达到最大重试次数（非 429 错误）
        error_msg = f"LLM API调用失败，已重试 {max_retries} 次"
        logger.error(f"{error_msg} - URL: {url}")
        return {
            "status": "error",
            "title_indices": [],
            "content_indices": [],
            "message": "",
            "error": error_msg
        }


class ZJCrawler:
    """浙江省政府网站爬虫"""

    def __init__(self, llm_api_key: str, max_concurrent_pages: int = 5):
        debug_log("初始化ZJCrawler")
        disable_system_proxies()
        self.llm_client = LLMApiClient(llm_api_key)
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.max_concurrent_pages = max_concurrent_pages
        self.request_count = 0  # 请求计数器
        
        # 创建持久化目录用于存储缓存
        self.cache_dir = Path("browser_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # 创建结果保存目录
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
        # 进度文件路径
        self.progress_file = Path("crawl_progress.json")
        
        # URL缓存：缓存已处理的URL，避免重复加载CSV
        # 格式: {(city_name, keyword): set(urls)}
        self.url_cache: Dict[Tuple[str, str], set] = {}

        self.zj_cities = {
            "杭州市": {"code": "3301", "websiteid": "330100000000000", "sitecode": "330101000000"},
            "宁波市": {"code": "3302", "websiteid": "330200000000000", "sitecode": "330201000000"},
            "温州市": {"code": "3303", "websiteid": "330300000000000", "sitecode": "330301000000"},
            "嘉兴市": {"code": "3304", "websiteid": "330400000000000", "sitecode": "330401000000"},
            "湖州市": {"code": "3305", "websiteid": "330500000000000", "sitecode": "330501000000"},
            "绍兴市": {"code": "3306", "websiteid": "330600000000000", "sitecode": "330601000000"},
            "金华市": {"code": "3307", "websiteid": "330700000000000", "sitecode": "330701000000"},
            "衢州市": {"code": "3308", "websiteid": "330800000000000", "sitecode": "330801000000"},
            "舟山市": {"code": "3309", "websiteid": "330900000000000", "sitecode": "330901000000"},
            "台州市": {"code": "3310", "websiteid": "331000000000000", "sitecode": "331001000000"},
            "丽水市": {"code": "3311", "websiteid": "331100000000000", "sitecode": "331101000000"}
        }

        self.base_url = "https://search.zj.gov.cn/jsearchfront/search.do"

    def setup_network_logging(self, page: Page):
        """为页面设置网络请求日志监听"""
        
        def on_request(request: Request):
            self.request_count += 1
            network_logger.info(f"[请求 #{self.request_count}] {request.method} {request.url}")
            if request.post_data:
                network_logger.debug(f"  请求体: {request.post_data[:500]}...")
            headers = request.headers
            if headers:
                network_logger.debug(f"  请求头: {dict(list(headers.items())[:5])}...")
        
        def on_response(response: Response):
            status = response.status
            url = response.url
            status_text = response.status_text
            log_level = logging.WARNING if status >= 400 else logging.INFO
            network_logger.log(log_level, f"[响应] {status} {status_text} - {url}")
            if status >= 400:
                network_logger.warning(f"  响应头: {response.headers}")
        
        def on_request_failed(request: Request):
            network_logger.error(f"[请求失败] {request.method} {request.url}")
            if request.failure:
                network_logger.error(f"  失败原因: {request.failure}")
        
        page.on("request", on_request)
        page.on("response", on_response)
        page.on("requestfailed", on_request_failed)

    def load_target_cities(self, csv_file: str) -> List[str]:
        """从CSV文件加载313个目标城市列表"""
        try:
            df = pd.read_csv(csv_file)
            cities = df['city'].tolist() if 'city' in df.columns else df.iloc[:, 0].tolist()
            debug_log(f"加载城市列表: {cities}")
            logger.info(f"加载了{len(cities)}个目标城市")
            return cities
        except Exception as e:
            logger.error(f"加载城市列表失败: {e}")
            return []

    def build_search_url(self, city_info: Dict, keyword: str, page: int = 1) -> str:
        """构建搜索URL"""
        params = {
            "websiteid": city_info["websiteid"],
            "tpl": "1569",
            "word": "",
            "temporaryQ": "",
            "synonyms": "",
            "checkError": "1",
            "p": str(page),
            "q": keyword,
            "siteCode": city_info["sitecode"],
            "searchid": "",
            "isContains": "1",
            "cateid": "370"
        }

        param_string = "&".join([f"{k}={v}" for k, v in params.items()])
        url = f"{self.base_url}?{param_string}"
        debug_log(f"构建搜索URL: {url}")
        return url

    async def extract_search_results(self, page: Page) -> Tuple[List[SearchResult], int]:
        """提取搜索结果页面的信息"""
        results = []
        total_pages = 0

        try:
            debug_log(f"等待搜索结果列表加载，当前URL: {page.url}")
            
            # 首先检查是否出现"未能搜到内容"的提示
            await page.wait_for_timeout(2000)
            
            # 等待搜索结果容器出现
            await page.wait_for_selector('.comprehensive', timeout=15000)
            container_visible = await page.is_visible('.comprehensive')
            debug_log(f"搜索结果容器已出现，可见状态: {container_visible}")

            try:
                total_pages_element = await page.query_selector('.totalPage')
                if total_pages_element:
                    total_text = await total_pages_element.inner_text()
                    # 使用更安全的字符串查找方法
                    if '共' in total_text and '页' in total_text:
                        start_idx = total_text.find('共') + 1
                        end_idx = total_text.find('页', start_idx)
                        if start_idx > 0 and end_idx > start_idx:
                            page_text = total_text[start_idx:end_idx].strip()
                            # 提取数字部分
                            page_number = ''.join(c for c in page_text if c.isdigit())
                            if page_number:
                                total_pages = int(page_number)
                                debug_log(f"检测到总页数: {total_pages}")
            except Exception as e:
                debug_log(f"读取总页数异常: {e}")

            result_items = await page.query_selector_all('.comprehensiveItem')
            debug_log(f"搜索结果条数: {len(result_items)}")

            for idx, item in enumerate(result_items, start=1):
                try:
                    title_element = await item.query_selector('.titleWrapper a')
                    if not title_element:
                        debug_log(f"第{idx}条结果缺少标题元素")
                        continue

                    title = await title_element.inner_text()
                    url = await title_element.get_attribute('href')

                    summary_element = await item.query_selector('.newsDescribe a')
                    summary = await summary_element.inner_text() if summary_element else ""

                    source_time_element = await item.query_selector('.sourceTime')
                    source_time_text = await source_time_element.inner_text() if source_time_element else ""

                    column = ""
                    date = ""
                    if source_time_text:
                        debug_log(f"解析来源和时间: {source_time_text}")
                        # 使用更稳健的字符串解析方法，避免正则回溯
                        source_time_line = source_time_text.strip().replace('\n', ' ').replace('\r', ' ')
                        
                        # 查找来源信息
                        source_start = source_time_line.find('来源:')
                        if source_start != -1:
                            source_start += 3  # 跳过"来源:"
                            # 查找时间标记的位置
                            time_start = source_time_line.find('时间:', source_start)
                            if time_start != -1:
                                # 提取来源部分（来源: 到 时间: 之间）
                                source_full = source_time_line[source_start:time_start].strip()
                            else:
                                # 如果没有时间标记，取到字符串末尾
                                source_full = source_time_line[source_start:].strip()
                            
                            # 处理来源中的栏目信息（取 "-" 后面的部分）
                            if '-' in source_full:
                                dash_pos = source_full.rfind('-')  # 使用 rfind 取最后一个 "-"
                                column = source_full[dash_pos + 1:].strip()
                            debug_log(f"解析出来源栏目: {column}")
                        
                        # 查找时间信息
                        time_start = source_time_line.find('时间:')
                        if time_start != -1:
                            time_start += 3  # 跳过"时间:"
                            # 查找下一个空格或字符串结束
                            time_end = len(source_time_line)
                            for i, char in enumerate(source_time_line[time_start:], time_start):
                                if char.isspace():
                                    time_end = i
                                    break
                            date = source_time_line[time_start:time_end].strip()
                            debug_log(f"解析出时间: {date}")

                    if url and title:
                        if url.startswith('visit/link.do'):
                            url = urljoin(page.url, url)

                        result = SearchResult(
                            keyword="",
                            title=title,
                            url=url,
                            summary=summary,
                            date=date,
                            column=column
                        )
                        results.append(result)
                        debug_log(f"解析结果: {result}")
                except Exception as e:
                    logger.warning(f"提取单个搜索结果失败: {e}")
                    continue

        except Exception as e:
            try:
                page_html = await page.content()
                debug_log(f"提取搜索结果失败页面URL: {page.url}")
                debug_log(f"失败页面HTML长度: {len(page_html)}")
                
                # DEBUG模式下截图保存当前页面状态
                if DEBUG:
                    try:
                        # 创建调试目录
                        debug_dir = Path("debug_screenshots")
                        debug_dir.mkdir(exist_ok=True)
                        
                        # 生成截图文件名，包含时间戳
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        url_hash = str(abs(hash(page.url)))[:8]  # URL哈希的前8位
                        screenshot_filename = debug_dir / f"search_fail_{timestamp}_{url_hash}.png"
                        
                        # 截取完整页面
                        await page.screenshot(path=str(screenshot_filename), full_page=True)
                        debug_log(f"已保存调试截图: {screenshot_filename}")
                        logger.warning(f"搜索页面无结果，已保存截图到: {screenshot_filename}")
                    except Exception as screenshot_error:
                        debug_log(f"保存调试截图失败: {screenshot_error}")
                
                # 检查是否包含无结果提示
                # if "很抱歉！未能搜到您想要的内容" in page_html:
                #     debug_log("页面显示无搜索结果")
                #     return results, total_pages
                    
                # debug_log(f"失败页面HTML关键部分: {page_html[page_html.find('<body'):page_html.find('</body>') + 7] if '<body' in page_html else page_html[:1000]}")
            except Exception as inner_exc:
                debug_log(f"获取失败页面HTML时出错: {inner_exc}")
            logger.error(f"提取搜索结果失败: {e}")

        return results, total_pages

    async def get_page_content(self, url: str, page: Page) -> str:
        """获取页面完整HTML内容，支持重试机制"""
        start_time = time.perf_counter()
        debug_log(f"开始获取页面内容: {url}")
        
        # 为当前页面设置网络日志
        self.setup_network_logging(page)
        
        max_retries = 10
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                if 'visit/link.do' in url:
                    debug_log(f"即将访问搜索页面 (第{retry_count + 1}次尝试): {url}")
                    await page.goto(url, wait_until='domcontentloaded', timeout=30000)
                    debug_log(f"搜索页面实际URL: {page.url}")
                    await page.wait_for_timeout(5000)
                    debug_log("搜索页面初步加载完成")

                actual_url = page.url
                if actual_url != url:
                    debug_log(f"检测到跳转至: {actual_url}")
                    await page.goto(actual_url, wait_until='domcontentloaded', timeout=30000)

                await page.wait_for_timeout(5000)
                html = await page.content()
                duration = time.perf_counter() - start_time
                debug_log(f"获取页面耗时: {duration:.2f}s, HTML长度: {len(html)}")

                if len(html) < 10:
                    if retry_count < max_retries:
                        retry_count += 2
                        logger.warning(f"页面内容非常短 (第{retry_count/2}次重试): {url}")
                        await asyncio.sleep(2)
                        continue
                    else:
                        logger.warning(f"页面内容非常短，重试{max_retries/2}次后仍失败: {url}")
                        return ""
                if len(html) < 500:
                    if retry_count < max_retries:
                        retry_count += 1
                        logger.warning(f"页面内容过短 (第{retry_count}次重试): {url}")
                        await asyncio.sleep(2)
                        continue
                    else:
                        logger.warning(f"页面内容过短，重试{max_retries}次后仍失败: {url}")
                        return ""

                return html

            except asyncio.TimeoutError:
                if retry_count < max_retries:
                    retry_count += 1
                    logger.warning(f"页面访问超时 (第{retry_count}次重试): {url}")
                    await asyncio.sleep(3)
                    continue
                else:
                    logger.error(f"页面访问超时，重试{max_retries}次后仍失败: {url}")
                    return ""
            except Exception as e:
                if retry_count < max_retries:
                    retry_count += 1
                    logger.warning(f"获取页面内容失败 (第{retry_count}次重试) {url}: {e}")
                    await asyncio.sleep(3)
                    continue
                else:
                    logger.error(f"获取页面内容失败，重试{max_retries}次后仍失败 {url}: {e}")
                    return ""
        
        return ""

    async def save_single_result_to_csv(self, city_name: str, keyword: str, result: SearchResult, page_num: int = 1):
        """保存单个结果到CSV文件"""
        debug_log(f"准备保存单个结果: {city_name} - {keyword} - 第{page_num}页")
        
        # 在results目录下创建城市文件夹
        city_folder = self.results_dir / city_name
        city_folder.mkdir(exist_ok=True)

        filename = city_folder / f"{city_name}-{keyword}-news.csv"

        # 准备数据
        data = {
            'keyword': self.sanitize_csv_content(result.keyword),
            'title': self.sanitize_csv_content(result.title),
            'url': self.sanitize_csv_content(result.url),
            'summary': self.sanitize_csv_content(result.summary),
            'date': self.sanitize_csv_content(result.date),
            'content': self.sanitize_csv_content(result.content),
            'column': self.sanitize_csv_content(result.column),
            'error': self.sanitize_csv_content(result.error),
            'page_num': page_num
        }

        df = pd.DataFrame([data])
        
        # CSV写入参数
        csv_params = {
            'index': False, 
            'encoding': 'utf-8',
            'quoting': 1,  # csv.QUOTE_ALL - 对所有字段加引号
            'escapechar': '\\',  # 转义字符
            'doublequote': False  # 不使用双引号转义
        }
        
        # 判断文件是否存在以决定是否写入表头
        file_exists = filename.exists()
        
        if not file_exists:
            # 文件不存在，创建新文件并写入表头
            df.to_csv(filename, mode='w', header=True, **csv_params)
            debug_log(f"创建新CSV文件并写入表头: {filename}")
        else:
            # 文件存在，追加模式写入，不写入表头
            df.to_csv(filename, mode='a', header=False, **csv_params)
            debug_log(f"追加到现有CSV文件: {filename}")

        # 记录保存状态
        if result.error:
            logger.info(f"已保存结果到: {filename} (错误: {result.error[:50]}...)")
        else:
            logger.info(f"已保存结果到: {filename} (成功)")

    def extract_numbered_chinese_blocks(self, html: str, max_blocks: int = 200) -> List[Tuple[int, str]]:
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(['script', 'style', 'noscript', 'link', 'iframe', 'svg', 'canvas', 'meta', 'head']):
            tag.decompose()
        
        # 使用函数检查中文字符，避免正则表达式
        def contains_chinese(text: str) -> bool:
            for char in text:
                if '\u4e00' <= char <= '\u9fff':
                    return True
            return False
        
        blocks: List[Tuple[int, str]] = []
        for text in soup.stripped_strings:
            # 使用字符串方法替代正则表达式压缩空白
            parts = text.strip().split()
            candidate = ' '.join(parts)
            
            if len(candidate) < 2:
                continue
            if not contains_chinese(candidate):
                continue
            blocks.append((len(blocks) + 1, candidate))
            if len(blocks) >= max_blocks:
                break
        debug_log(f"提取到中文文本块 {len(blocks)} 个")
        return blocks

    async def process_single_result(self, result: SearchResult, semaphore: asyncio.Semaphore, 
                                     existing_urls: set, city_name: str, keyword: str) -> SearchResult:
        """处理单个搜索结果，获取内容并使用选择器提取"""
        # 检查URL是否已存在
        if result.url in existing_urls:
            result.error = "URL已存在，跳过处理"
            logger.info(f"跳过已存在的URL: {result.url}")
            return result
        
        async with semaphore:
            page = None
            try:
                if not result.url:
                    result.error = "URL为空"
                    return result

                debug_log(f"处理结果: {result.url}")
                page = await self.context.new_page()
                
                # 获取页面HTML内容（已包含重试逻辑）
                html = await self.get_page_content(result.url, page)
                
                # 检查HTML是否有效
                if not html or len(html.strip()) < 500:
                    result.error = "页面无法访问或内容为空"
                    debug_log(f"跳过处理，页面内容无效: {result.url}")
                    return result
                
                # 预检查HTML中是否包含中文内容（避免正则回溯）
                def contains_chinese(text: str) -> bool:
                    for char in text:
                        if '\u4e00' <= char <= '\u9fff':
                            return True
                    return False
                
                if not contains_chinese(html):
                    result.error = "页面不包含中文内容"
                    debug_log(f"跳过处理，页面无中文内容: {result.url}")
                    return result

                blocks = self.extract_numbered_chinese_blocks(html)
                if not blocks:
                    result.error = "页面未提取到中文文本块"
                    debug_log(f"跳过处理，未提取到文本块: {result.url}")
                    return result

                # LLM任务重试配置
                max_llm_retries = 3  # LLM任务最多重试3次
                llm_retry_count = 0
                llm_result = None
                
                # 重试循环：针对LLM相关错误
                while llm_retry_count <= max_llm_retries:
                    llm_result = await self.llm_client.select_text_blocks(blocks, result.url)
                    
                    # 检查LLM返回状态
                    if llm_result.get("status") != "success":
                        llm_error = llm_result.get("error") or llm_result.get("message") or "LLM未返回成功状态"
                        
                        if llm_retry_count < max_llm_retries:
                            llm_retry_count += 1
                            logger.warning(f"LLM分析失败，重新提交任务 (第{llm_retry_count}/{max_llm_retries}次): {llm_error}")
                            logger.warning(f"URL: {result.url}")
                            await asyncio.sleep(1)
                            continue
                        else:
                            result.error = llm_error
                            debug_log(f"LLM分析失败，已重试{max_llm_retries}次: {result.error}")
                            return result
                    
                    # 解析索引
                    block_map = {idx: text for idx, text in blocks}

                    def normalize_indices(raw_indices):
                        """解析索引，支持区间表示法"""
                        normalized = []
                        for item in raw_indices or []:
                            # 处理区间 [start, end]
                            if isinstance(item, list) and len(item) == 2:
                                try:
                                    start, end = int(item[0]), int(item[1])
                                    for idx in range(start, end + 1):
                                        if idx in block_map and idx not in normalized:
                                            normalized.append(idx)
                                except (TypeError, ValueError) as e:
                                    debug_log(f"解析区间失败 {item}: {e}")
                                    continue
                            # 处理单个编号
                            else:
                                try:
                                    value = int(item)
                                    if value in block_map and value not in normalized:
                                        normalized.append(value)
                                except (TypeError, ValueError) as e:
                                    debug_log(f"解析单个编号失败 {item}: {e}")
                                    continue
                        return normalized

                    title_indices = normalize_indices(llm_result.get("title_indices"))
                    content_indices = normalize_indices(llm_result.get("content_indices"))

                    debug_log(f"标题编号: {title_indices}, 正文编号: {content_indices}")

                    # 检查是否有正文编号
                    if not content_indices:
                        if llm_retry_count < max_llm_retries:
                            llm_retry_count += 1
                            logger.warning(f"LLM未返回正文编号，重新提交任务 (第{llm_retry_count}/{max_llm_retries}次)")
                            logger.warning(f"URL: {result.url}")
                            logger.warning(f"LLM返回的content_indices: {llm_result.get('content_indices')}")
                            await asyncio.sleep(1)
                            continue
                        else:
                            result.error = "LLM未返回正文编号"
                            debug_log(f"LLM未返回正文编号，已重试{max_llm_retries}次")
                            return result
                    
                    # 成功获取到正文编号，跳出重试循环
                    break
                
                # 提取标题和正文
                title_text = " ".join(block_map[idx] for idx in title_indices).strip() if title_indices else ""
                if title_text:
                    result.title = title_text

                content_segments = [block_map[idx] for idx in content_indices]
                result.content = "\n\n".join(content_segments).strip()

                if not result.content:
                    result.error = "提取的正文为空"
                    debug_log(f"提取的正文为空: {result.url}")
                # elif len(result.content) < 100:
                #     result.error = "提取的正文过短"
                else:
                    debug_log(f"成功提取正文，长度 {len(result.content)} 字符，选中片段 {len(content_segments)} 个")
                    result.error = ""
                    # 成功处理后，将URL添加到缓存
                    self.add_url_to_cache(city_name, keyword, result.url)
                    
            except asyncio.TimeoutError:
                error_msg = f"处理超时: {result.url}"
                logger.error(error_msg)
                result.error = error_msg
            except Exception as e:
                error_msg = f"处理页面失败: {str(e)}"
                logger.error(f"{error_msg} - URL: {result.url}")
                result.error = error_msg
            finally:
                if page:
                    await page.close()
            
        return result

    async def process_and_save_single_result(
        self,
        result: SearchResult, 
        semaphore: asyncio.Semaphore,
        existing_urls: set,
        city_name: str,
        keyword: str,
        page_num: int
    ) -> SearchResult:
        """处理单个结果并立即保存（用于生产者-消费者模式）"""
        # 检查URL是否已存在
        if result.url in existing_urls:
            result.error = "URL已存在，跳过处理"
            debug_log(f"跳过已存在的URL: {result.url}")
            return result
        
        # 处理结果
        processed = await self.process_single_result(
            result, semaphore, existing_urls, city_name, keyword
        )
        
        # 立即保存（非跳过的结果）
        if processed.error != "URL已存在，跳过处理":
            await self.save_single_result_to_csv(city_name, keyword, processed, page_num)
        
        return processed

    async def fetch_search_page(
        self, 
        city_info: Dict, 
        keyword: str, 
        page_num: int
    ) -> Tuple[List[SearchResult], int]:
        """获取指定页的搜索结果（独立方法，用于生产者）"""
        url = self.build_search_url(city_info, keyword, page_num)
        city_name = [name for name, info in self.zj_cities.items() if info == city_info][0] if city_info in self.zj_cities.values() else "未知城市"
        
        logger.info(f"正在爬取搜索页: {city_name} - {keyword} - 第{page_num}页")
        network_logger.info(f"===== 开始爬取: {city_name} - {keyword} - 第{page_num}页 =====")
        
        # 创建新页面用于搜索
        page = await self.context.new_page()
        self.setup_network_logging(page)
        
        results = []
        total_pages = 0
        retry_count = 0
        max_retries = 10
        
        try:
            while retry_count < max_retries:
                try:
                    debug_log(f"尝试访问搜索页 (第{retry_count + 1}次): {url}")
                    
                    await page.set_extra_http_headers({
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
                        "Accept-Encoding": "gzip, deflate, br",
                        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                        "Cache-Control": "max-age=0",
                        "Connection": "keep-alive",
                        "DNT": "1",
                        "Sec-Fetch-Dest": "document",
                        "Sec-Fetch-Mode": "navigate",
                        "Sec-Fetch-Site": "none",
                        "Sec-Fetch-User": "?1",
                        "Upgrade-Insecure-Requests": "1",
                        "sec-ch-ua": '"Google Chrome";v="124", "Chromium";v="124", "Not-A.Brand";v="99"',
                        "sec-ch-ua-mobile": "?0",
                        "sec-ch-ua-platform": '"Windows"'
                    })
                    
                    await page.goto(url, wait_until='networkidle', timeout=30000)
                    await page.wait_for_timeout(3000)
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    await page.wait_for_timeout(1000)
                    await page.evaluate("window.scrollTo(0, 0)")
                    await page.wait_for_timeout(1000)

                    results, total_pages = await self.extract_search_results(page)
                    
                    if not results and retry_count < max_retries - 1:
                        debug_log(f"第{retry_count + 1}次尝试无结果，{2}秒后重试")
                        await asyncio.sleep(2)
                        retry_count += 1
                        continue
                    else:
                        break

                except Exception as e:
                    retry_count += 1
                    error_msg = f"访问搜索页失败 (第{retry_count}次): {str(e)}"
                    logger.warning(error_msg)
                    
                    if retry_count >= max_retries:
                        logger.error(f"重试{max_retries}次后仍然失败: {url}")
                        break
                    else:
                        await asyncio.sleep(3)
            
            # 设置关键词
            for result in results:
                result.keyword = keyword
            
            logger.info(f"搜索页第{page_num}页获取完成: {len(results)}条结果")
            return results, total_pages
            
        finally:
            await page.close()

    async def process_single_city_keyword(self, city_name: str, city_info: Dict,
                                          keyword: str, session: aiohttp.ClientSession) -> List[SearchResult]:
        """处理单个城市的单个关键词搜索 - 并行爬取版本（优化搜索页并行）"""
        all_results = []
        
        # 加载已存在的URL（只加载一次，后续使用缓存）
        existing_urls = self.load_existing_urls(city_name, keyword)
        debug_log(f"开始处理城市 {city_name} 的关键词 {keyword}，已有URL数量: {len(existing_urls)}")

        # 结果队列：存储待处理的搜索结果
        # 格式: (page_num, SearchResult)
        results_queue: asyncio.Queue = asyncio.Queue()
        
        # 搜索页爬取状态
        search_state = {
            'max_pages': None,
            'search_finished': False,
            'pages_fetched': set(),  # 已爬取的页码
        }
        
        # 创建信号量限制搜索页的并发数
        search_page_semaphore = asyncio.Semaphore(3)  # 最多3个搜索页并行

        async def fetch_single_search_page(page_num: int):
            """获取单个搜索页并将结果放入队列"""
            async with search_page_semaphore:
                try:
                    # 检查是否已爬取
                    if page_num in search_state['pages_fetched']:
                        debug_log(f"跳过已爬取的搜索页: 第{page_num}页")
                        return
                    
                    results, total_pages = await self.fetch_search_page(
                        city_info, keyword, page_num
                    )
                    
                    # 标记为已爬取
                    search_state['pages_fetched'].add(page_num)
                    
                    # 更新最大页数
                    if search_state['max_pages'] is None and total_pages > 0:
                        search_state['max_pages'] = total_pages
                        logger.info(f"{city_name} - {keyword}: 共{total_pages}页")
                    
                    # 如果没有结果，记录但不中断其他页
                    if not results:
                        logger.warning(f"第{page_num}页无搜索结果")
                        return
                    
                    # 将结果放入队列
                    for result in results:
                        await results_queue.put((page_num, result))
                    
                    logger.info(f"已将第{page_num}页的 {len(results)} 个结果加入队列")
                    
                except Exception as e:
                    logger.error(f"爬取搜索页第{page_num}页失败: {e}")

        async def search_page_producer():
            """生产者：并行爬取多个搜索页"""
            try:
                # 先爬取第一页获取总页数
                await fetch_single_search_page(1)
                await asyncio.sleep(1)  # 短暂延迟
                
                # 如果有总页数，启动并行爬取
                if search_state['max_pages']:
                    # 创建并行任务列表
                    tasks = []
                    for page_num in range(2, search_state['max_pages'] + 1):
                        task = asyncio.create_task(fetch_single_search_page(page_num))
                        tasks.append(task)
                        
                        # 每启动3个任务后短暂延迟，避免过载
                        if len(tasks) % 3 == 0:
                            await asyncio.sleep(0.5)
                    
                    # 等待所有搜索页爬取完成
                    if tasks:
                        await asyncio.gather(*tasks, return_exceptions=True)
                
                search_state['search_finished'] = True
                logger.info("搜索页生产者任务完成")
                
            except Exception as e:
                logger.error(f"搜索页生产者异常: {e}")
                import traceback
                logger.error(traceback.format_exc())
                search_state['search_finished'] = True
            finally:
                # 发送完成信号
                await results_queue.put((None, None))

        async def result_consumer():
            """消费者：从队列中取出结果并处理"""
            # 创建信号量限制并发
            semaphore = asyncio.Semaphore(self.max_concurrent_pages)
            active_tasks = []
            
            try:
                while True:
                    # 从队列获取结果
                    page_num, result = await results_queue.get()
                    
                    # 收到完成信号
                    if page_num is None and result is None:
                        logger.info("收到搜索完成信号，等待剩余任务完成...")
                        # 等待所有活跃任务完成
                        if active_tasks:
                            completed_results = await asyncio.gather(*active_tasks, return_exceptions=True)
                            # 收集结果
                            for res in completed_results:
                                if isinstance(res, SearchResult):
                                    all_results.append(res)
                                elif isinstance(res, Exception):
                                    logger.error(f"处理结果任务异常: {res}")
                        break
                    
                    # 创建处理任务
                    task = asyncio.create_task(
                        self.process_and_save_single_result(
                            result, semaphore, existing_urls, 
                            city_name, keyword, page_num
                        )
                    )
                    active_tasks.append(task)
                    
                    # 清理已完成的任务并收集结果
                    done_tasks = [t for t in active_tasks if t.done()]
                    for task in done_tasks:
                        try:
                            processed_result = await task
                            all_results.append(processed_result)
                        except Exception as e:
                            logger.error(f"处理结果任务异常: {e}")
                    
                    # 保留未完成的任务
                    active_tasks = [t for t in active_tasks if not t.done()]
                    
                    # 如果活跃任务达到上限，等待至少一个完成
                    if len(active_tasks) >= self.max_concurrent_pages:
                        done, pending = await asyncio.wait(
                            active_tasks, 
                            return_when=asyncio.FIRST_COMPLETED
                        )
                        
                        # 收集已完成任务的结果
                        for task in done:
                            try:
                                processed_result = await task
                                all_results.append(processed_result)
                            except Exception as e:
                                logger.error(f"处理结果任务异常: {e}")
                        
                        active_tasks = list(pending)
                    
                    # 标记队列任务完成
                    results_queue.task_done()
                
                logger.info("结果消费者任务完成")
                
            except Exception as e:
                logger.error(f"消费者任务异常: {e}")
                import traceback
                logger.error(traceback.format_exc())

        # 启动生产者和消费者
        producer_task = asyncio.create_task(search_page_producer())
        consumer_task = asyncio.create_task(result_consumer())
        
        try:
            # 等待两个任务完成
            await asyncio.gather(producer_task, consumer_task)
            
        except Exception as e:
            logger.error(f"并行爬取任务异常: {e}")
            import traceback
            logger.error(traceback.format_exc())

        logger.info(f"完成 {city_name} - {keyword}: 总共{len(all_results)}条结果，已逐个保存")
        network_logger.info(f"===== 完成 {city_name} - {keyword}: 总共{len(all_results)}条结果 =====")
        return all_results

    def sanitize_csv_content(self, content: str) -> str:
        """
        清理CSV内容中的特殊字符，防止破坏CSV结构
        """
        if not content:
            return content
            
        # 转义常见的控制字符
        content = content.replace('\n', '\\n')
        content = content.replace('\r', '\\r')
        content = content.replace('\t', '\\t')
        content = content.replace('"', '\\"')
        content = content.replace("'", "\\'")
        
        # 移除控制字符（避免正则表达式）
        cleaned = []
        for char in content:
            code = ord(char)
            # 跳过控制字符: 0x00-0x08, 0x0B, 0x0C, 0x0E-0x1F, 0x7F
            if (0x00 <= code <= 0x08) or code == 0x0B or code == 0x0C or \
               (0x0E <= code <= 0x1F) or code == 0x7F:
                continue
            cleaned.append(char)
        
        content = ''.join(cleaned)
        
        # 去除首尾空白字符
        content = content.strip()
        
        return content

    def load_llm_api_key(self) -> str:
        """从文件中加载LLM API key，支持跨平台路径"""
        system_type = platform.system().lower()
        
        if system_type == "windows":
            # Windows系统：使用用户主目录
            api_key_file = Path.home() / ".siliconflow_apikey"
        else:
            # Linux/macOS系统：使用用户主目录
            api_key_file = Path.home() / ".siliconflow_apikey"
            
        debug_log(f"检测到操作系统: {system_type}")
        debug_log(f"API key文件路径: {api_key_file}")
        
        try:
            with open(api_key_file, 'r', encoding='utf-8') as f:
                api_key = f.read().strip()
            if not api_key:
                raise ValueError("API key为空")
            debug_log(f"成功从文件加载API key: {api_key_file}")
            return api_key
        except FileNotFoundError:
            logger.error(f"API key文件不存在: {api_key_file}")
            logger.error(f"请在以下位置创建API key文件:")
            if system_type == "windows":
                logger.error(f"  Windows: {api_key_file}")
                logger.error(f"  示例: echo your-api-key > {api_key_file}")
            else:
                logger.error(f"  Linux/macOS: {api_key_file}")
                logger.error(f"  示例: echo 'your-api-key' > {api_key_file}")
            raise
        except Exception as e:
            logger.error(f"读取API key失败: {e}")
            raise

    def load_progress(self) -> Dict[str, Dict[str, bool]]:
        """加载爬取进度"""
        if not self.progress_file.exists():
            logger.info("未找到进度文件，将从头开始爬取")
            return {}
        
        try:
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                progress_data = json.load(f)
            logger.info(f"已加载进度文件，包含 {len(progress_data)} 个源城市的记录")
            return progress_data
        except Exception as e:
            logger.error(f"加载进度文件失败: {e}，将从头开始爬取")
            return {}

    def save_progress(self, progress: Dict[str, Dict[str, bool]]):
        """保存爬取进度"""
        try:
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress, f, ensure_ascii=False, indent=2)
            debug_log(f"已保存进度到: {self.progress_file}")
        except Exception as e:
            logger.error(f"保存进度文件失败: {e}")

    def is_city_pair_completed(self, progress: Dict[str, Dict[str, bool]], 
                                source_city: str, target_city: str) -> bool:
        """检查城市对是否已完成"""
        if source_city not in progress:
            return False
        return progress[source_city].get(target_city, False)

    def mark_city_pair_completed(self, progress: Dict[str, Dict[str, bool]], 
                                  source_city: str, target_city: str):
        """标记城市对为已完成"""
        if source_city not in progress:
            progress[source_city] = {}
        progress[source_city][target_city] = True
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"标记完成: {source_city} -> {target_city} (时间: {timestamp})")
        # 立即保存进度
        self.save_progress(progress)

    def load_existing_urls(self, city_name: str, keyword: str) -> set:
        """从CSV文件加载已存在的URL集合，使用缓存优化性能"""
        cache_key = (city_name, keyword)
        
        # 检查缓存
        if cache_key in self.url_cache:
            debug_log(f"从缓存获取URL集合: {city_name}-{keyword}, 数量: {len(self.url_cache[cache_key])}")
            return self.url_cache[cache_key]
        
        # 缓存未命中，从CSV加载
        city_folder = self.results_dir / city_name
        filename = city_folder / f"{city_name}-{keyword}-news.csv"
        
        if not filename.exists():
            debug_log(f"CSV文件不存在，创建空URL集合: {filename}")
            self.url_cache[cache_key] = set()
            return self.url_cache[cache_key]
        
        try:
            # 只读取url列，提高性能
            df = pd.read_csv(filename, encoding='utf-8', usecols=['url'])
            existing_urls = set(df['url'].dropna().tolist())
            
            # 存入缓存
            self.url_cache[cache_key] = existing_urls
            logger.info(f"从 {filename} 加载了 {len(existing_urls)} 个已存在的URL并缓存")
            return existing_urls
            
        except Exception as e:
            logger.error(f"加载已有URL失败 {filename}: {e}")
            self.url_cache[cache_key] = set()
            return self.url_cache[cache_key]

    def add_url_to_cache(self, city_name: str, keyword: str, url: str):
        """将新URL添加到缓存中"""
        cache_key = (city_name, keyword)
        if cache_key not in self.url_cache:
            self.url_cache[cache_key] = set()
        self.url_cache[cache_key].add(url)
        debug_log(f"添加URL到缓存: {url}")

    def clear_url_cache_for_pair(self, city_name: str, keyword: str):
        """清理特定城市对的URL缓存"""
        cache_key = (city_name, keyword)
        if cache_key in self.url_cache:
            del self.url_cache[cache_key]
            debug_log(f"已清理URL缓存: {city_name}-{keyword}")

    def cleanup_browser_cache(self):
        """清理浏览器缓存目录"""
        if self.cache_dir.exists():
            try:
                # 计算缓存目录大小
                total_size = sum(f.stat().st_size for f in self.cache_dir.rglob('*') if f.is_file())
                size_mb = total_size / (1024 * 1024)
                
                logger.info(f"开始清理浏览器缓存目录: {self.cache_dir}")
                logger.info(f"缓存目录大小: {size_mb:.2f} MB")
                
                # 删除目录内所有内容
                shutil.rmtree(self.cache_dir)
                
                # 重新创建空目录
                self.cache_dir.mkdir(exist_ok=True)
                
                logger.info(f"浏览器缓存已清理，释放空间: {size_mb:.2f} MB")
            except Exception as e:
                logger.error(f"清理浏览器缓存失败: {e}")

    async def run_crawler(self, cities_csv_file: str):
        """运行爬虫主程序"""
        logger.info("启动爬虫")
        network_logger.info("===== 爬虫启动，开始记录网络请求 =====")
        
        # 从文件加载API key
        try:
            llm_api_key = self.load_llm_api_key()
        except Exception as e:
            logger.error(f"无法加载API key: {e}")
            return
        
        target_cities = self.load_target_cities(cities_csv_file)
        if not target_cities:
            logger.error("无法加载目标城市列表")
            return

        # 加载爬取进度
        progress = self.load_progress()

        async with LLMApiClient(llm_api_key) as llm_client:
            self.llm_client = llm_client

            async with async_playwright() as p:
                # 使用更完整的浏览器配置
                self.browser = await p.chromium.launch(
                    headless=True,  
                    args=[
                        '--no-sandbox',
                        '--disable-setuid-sandbox',
                        '--disable-dev-shm-usage',
                        '--disable-accelerated-2d-canvas',
                        '--no-first-run',
                        '--no-zygote',
                        '--disable-gpu',
                        '--disable-background-timer-throttling',
                        '--disable-backgrounding-occluded-windows',
                        '--disable-renderer-backgrounding',
                        '--disable-features=TranslateUI',
                        '--disable-ipc-flooding-protection',
                        '--disable-web-security',
                        '--disable-features=VizDisplayCompositor',
                        # 启用缓存相关参数
                        f'--disk-cache-dir={self.cache_dir}',
                        '--disk-cache-size=209715200',  # 200MB 缓存
                        # 性能优化参数
                        '--disable-software-rasterizer',  # 禁用软件光栅化
                        '--disable-extensions',  # 禁用扩展
                        '--disable-plugins',  # 禁用插件
                        '--disable-images',  # 禁用图片加载（爬取文本不需要图片）
                        '--blink-settings=imagesEnabled=false',  # 确保图片禁用
                        '--disable-javascript-harmony-shipping',  # 禁用实验性JS特性
                    ],
                    proxy=None
                )
                debug_log("Chromium浏览器已启动（已启用磁盘缓存）")
                
                # 创建浏览器上下文，启用服务工作线程和离线缓存
                self.context = await self.browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                    viewport={"width": 1920, "height": 1080},
                    locale="zh-CN",
                    timezone_id="Asia/Shanghai",
                    java_script_enabled=True,
                    ignore_https_errors=True,
                    # 启用服务工作线程（Service Workers）
                    service_workers='allow',
                    extra_http_headers={
                        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                        "Upgrade-Insecure-Requests": "1",
                        "DNT": "1",
                        # 允许浏览器使用缓存
                        "Cache-Control": "max-age=3600",
                    }
                )
                debug_log(f"浏览器上下文已创建（启用缓存策略），最大并发页面数: {self.max_concurrent_pages}")

                try:
                    for city_name, city_info in self.zj_cities.items():
                        logger.info(f"开始处理城市: {city_name}")
                        network_logger.info(f"===== 开始处理城市: {city_name} =====")

                        async with aiohttp.ClientSession(trust_env=False) as session:
                            for target_city in target_cities:
                                target_city_name = str(target_city).strip()
                                if not target_city_name:
                                    debug_log(f"跳过空白城市记录: {target_city}")
                                    continue
                                if target_city_name == city_name:
                                    logger.info(f"跳过自身城市: {city_name}")
                                    continue
                                
                                # 检查是否已完成
                                if self.is_city_pair_completed(progress, city_name, target_city_name):
                                    logger.info(f"跳过已完成的城市对: {city_name} -> {target_city_name}")
                                    continue
                                
                                try:
                                    debug_log(f"开始任务: {city_name} -> {target_city_name}")
                                    results = await self.process_single_city_keyword(
                                        city_name, city_info, target_city_name, session
                                    )

                                    if results:
                                        logger.info(f"{city_name}-{target_city_name} 完成，共{len(results)}条结果已保存")
                                    else:
                                        logger.info(f"{city_name}-{target_city_name} 无有效结果")
                                    
                                    # 标记为已完成
                                    self.mark_city_pair_completed(progress, city_name, target_city_name)
                                    
                                    # 清理该城市对的URL缓存，释放内存
                                    self.clear_url_cache_for_pair(city_name, target_city_name)

                                    await asyncio.sleep(3)  # 增加延时

                                except Exception as e:
                                    logger.error(f"处理 {city_name}-{target_city_name} 失败: {e}")
                                    # 不标记为完成，下次可以重试
                                    # 清理缓存，下次重新加载
                                    self.clear_url_cache_for_pair(city_name, target_city_name)
                                    continue

                        logger.info(f"完成城市: {city_name}")

                        network_logger.info(f"===== 完成城市: {city_name} =====")

                finally:
                    if self.context:
                        await self.context.close()
                        self.context = None
                        debug_log("Chromium浏览器上下文已关闭")
                    if self.browser:
                        await self.browser.close()
                        debug_log("Chromium浏览器已关闭")
                    
                    # 清理浏览器缓存
                    self.cleanup_browser_cache()
                    
                    network_logger.info("===== 爬虫结束，网络请求记录完毕 =====")


async def main():
    """主函数"""
    cities_csv_file = "target_cities.csv"

    # 不再需要手动提供API key，直接从文件读取
    crawler = ZJCrawler("", max_concurrent_pages=120)  # API key会从文件自动加载
    await crawler.run_crawler(cities_csv_file)


if __name__ == "__main__":
    asyncio.run(main())