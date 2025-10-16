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
from typing import Any, Dict, List, Tuple, Optional, Set
from urllib.parse import urljoin, urlparse
import pandas as pd
from playwright.async_api import async_playwright, Page, Browser, BrowserContext, Request, Response
import aiohttp
import logging
from dataclasses import dataclass, field
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
    page_num: int = 1
    search_page_title: str = ""  


@dataclass
class SearchTask:
    """搜索任务"""
    city_name: str
    city_info: Dict
    keyword: str
    page_num: int


@dataclass
class ContentTask:
    """内容提取任务"""
    result: SearchResult
    city_name: str
    keyword: str
    existing_urls: Set[str]


@dataclass
class AnalysisTask:
    """LLM分析任务"""
    result: SearchResult
    blocks: List[Tuple[int, str]]
    city_name: str
    keyword: str


@dataclass
class SaveTask:
    """保存任务"""
    result: SearchResult
    city_name: str
    keyword: str


# 应用程序的主要逻辑类
class SearchResultFetcher:
    """搜索结果获取器 - 负责从搜索页获取结果列表"""
    
    def __init__(self, context: BrowserContext, base_url: str):
        self.context = context
        self.base_url = base_url
        self.request_count = 0
    
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
        return f"{self.base_url}?{param_string}"
    
    async def extract_search_results(self, page: Page) -> Tuple[List[SearchResult], int]:
        """提取搜索结果页面的信息"""
        results = []
        total_pages = 0

        try:
            await page.wait_for_selector('.comprehensive', timeout=15000)

            try:
                total_pages_element = await page.query_selector('.totalPage')
                if total_pages_element:
                    total_text = await total_pages_element.inner_text()
                    if '共' in total_text and '页' in total_text:
                        start_idx = total_text.find('共') + 1
                        end_idx = total_text.find('页', start_idx)
                        if start_idx > 0 and end_idx > start_idx:
                            page_text = total_text[start_idx:end_idx].strip()
                            page_number = ''.join(c for c in page_text if c.isdigit())
                            if page_number:
                                total_pages = int(page_number)
            except Exception as e:
                debug_log(f"读取总页数异常: {e}")

            result_items = await page.query_selector_all('.comprehensiveItem')

            for idx, item in enumerate(result_items, start=1):
                try:
                    title_element = await item.query_selector('.titleWrapper a')
                    if not title_element:
                        logger.warning(f"提取结果标题失败: {page.url}")
                        continue

                    # 优先使用title属性获取标题
                    title = await title_element.get_attribute('title')
                    if not title:
                        # 如果title属性为空，则使用链接文本内容
                        title = await title_element.inner_text()
                    
                    url = await title_element.get_attribute('href')

                    summary_element = await item.query_selector('.newsDescribe a')
                    summary = await summary_element.inner_text() if summary_element else ""

                    source_time_element = await item.query_selector('.sourceTime')
                    source_time_text = await source_time_element.inner_text() if source_time_element else ""

                    column = ""
                    date = ""
                    if source_time_text:
                        source_time_line = source_time_text.strip().replace('\n', ' ').replace('\r', ' ')
                        
                        source_start = source_time_line.find('来源:')
                        if source_start != -1:
                            source_start += 3
                            time_start = source_time_line.find('时间:', source_start)
                            if time_start != -1:
                                source_full = source_time_line[source_start:time_start].strip()
                            else:
                                source_full = source_time_line[source_start:].strip()
                            
                            if '-' in source_full:
                                dash_pos = source_full.rfind('-')
                                column = source_full[dash_pos + 1:].strip()
                        
                        time_start = source_time_line.find('时间:')
                        if time_start != -1:
                            time_start += 3
                            time_end = len(source_time_line)
                            for i, char in enumerate(source_time_line[time_start:], time_start):
                                if char.isspace():
                                    time_end = i
                                    break
                            date = source_time_line[time_start:time_end].strip()

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
                except Exception as e:
                    logger.warning(f"提取单个搜索结果失败: {e}")
                    continue

        except Exception as e:
            logger.error(f"提取搜索结果失败: {e}")

        return results, total_pages
    
    async def fetch_search_page(self, task: SearchTask) -> Tuple[List[SearchResult], int]:
        """执行单个搜索任务"""
        url = self.build_search_url(task.city_info, task.keyword, task.page_num)
        
        logger.info(f"正在爬取搜索页: {task.city_name} - {task.keyword} - 第{task.page_num}页")
        
        page = await self.context.new_page()
        
        results = []
        total_pages = 0
        retry_count = 0
        max_retries = 10
        
        try:
            while retry_count < max_retries:
                try:
                    await page.set_extra_http_headers({
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                        "Accept-Language": "zh-CN,zh;q=0.9",
                    })
                    
                    await page.goto(url, wait_until='networkidle', timeout=30000)

                    results, total_pages = await self.extract_search_results(page)
                    
                    if not results and retry_count < max_retries - 1:
                        await asyncio.sleep(2)
                        retry_count += 1
                        continue
                    else:
                        break

                except Exception as e:
                    retry_count += 1
                    logger.warning(f"访问搜索页失败 (第{retry_count}次): {str(e)}")
                    
                    if retry_count >= max_retries:
                        logger.error(f"重试{max_retries}次后仍然失败: {url}")
                        break
                    else:
                        await asyncio.sleep(3)
            
            for result in results:
                result.keyword = task.keyword
                result.page_num = task.page_num
            
            return results, total_pages
            
        finally:
            await page.close()


class LLMApiClient:
    """LLM API客户端，使用OpenAI SDK调用SiliconFlow服务"""

    # 可用模型列表，按优先级排序
    AVAILABLE_MODELS = ["models/qwen3-14b-awq"] if USE_LOCAL_VLLM else [
        "Qwen/Qwen3-8B",
        "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        "Qwen/Qwen2.5-7B-Instruct",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    ]

    def __init__(self, api_key: str, base_url: str = "http://localhost:8000/v1" if USE_LOCAL_VLLM else "https://api.siliconflow.cn/"):
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.current_model_index = 0
        self.model_usage_count = {model: 0 for model in self.AVAILABLE_MODELS}
        self.model_switch_lock = asyncio.Lock()
        self.last_429_model = None
        self.model_429_count = {model: 0 for model in self.AVAILABLE_MODELS}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        logger.info("模型使用统计:")
        for model, count in self.model_usage_count.items():
            logger.info(f"  {model}: {count} 次")
        
        logger.info("模型429错误统计:")
        for model, count in self.model_429_count.items():
            if count > 0:
                logger.info(f"  {model}: {count} 次")
        
        return False

    def get_current_model(self) -> str:
        """获取当前使用的模型"""
        return self.AVAILABLE_MODELS[self.current_model_index]

    async def switch_model(self):
        """切换到下一个可用模型"""
        async with self.model_switch_lock:
            self.current_model_index = (self.current_model_index + 1) % len(self.AVAILABLE_MODELS)
            logger.info(f"切换到新模型: {self.get_current_model()}")

    async def handle_rate_limit_error(self, current_model: str) -> str:
        """处理速率限制错误，尝试切换模型或等待"""
        if current_model == self.last_429_model:
            # 连续两次429错误，切换模型
            await self.switch_model()
        else:
            self.last_429_model = current_model
        
        return self.get_current_model()

    def extract_json_from_response(self, response: str) -> str:
        """从LLM的响应中提取JSON字符串"""
        try:
            # 尝试直接解析为JSON
            json_obj = json.loads(response)
            return json.dumps(json_obj, ensure_ascii=False)
        except json.JSONDecodeError:
            pass
        
        # 如果直接解析失败，尝试提取可能的JSON片段
        json_pattern = re.compile(r'({.*?})', re.S)
        matches = json_pattern.findall(response)
        
        if matches:
            # 返回第一个匹配的JSON片段
            return matches[0]
        
        return ""

    def fix_json_format(self, json_str: str) -> str:
        """修复JSON字符串的格式问题"""
        # 替换单引号为双引号
        json_str = json_str.replace("'", '"')
        
        # 修复缺失的逗号
        json_str = re.sub(r'([a-zA-Z0-9}])\s*([a-zA-Z0-9{])', r'\1,\2', json_str)
        
        return json_str

    async def select_text_blocks(self, blocks: List[Tuple[int, str]], url: str) -> Dict[str, Any]:
        """选择文本块"""
        prompt = f"""请分析以下网页文本片段，找出其中的**正文**。

输出要求：
1. `status`： `"success"` 或 `"error"`
1. `content_indices`：正文的编号，可能包含多个编号或区间，按顺序排列
4. `message`：错误原因

你可以使用一个二维数组来表示编号和区间，例如[[1,3], 8, [15,45]]表示编号1到3，编号8，以及编号15到45。

输出格式要求：
一个合法的 JSON 对象：
{{"status":"success或error","content_indices":编号, "message":"错误原因"}}

待分析的文本片段：

"""
        
        for idx, text in blocks:
            prompt += f"{idx}. {text}\n"

        max_retries = 10
        retry_count = 0
        rate_limit_cycle_count = 0
        json_parse_retry = 0
        max_json_retries = 5

        while retry_count < max_retries:
            current_model = self.get_current_model()
            
            try:
                logger.info(f"使用模型: {current_model} - URL: {url[:80]}...")
                
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
                    timeout=30,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}} if USE_LOCAL_VLLM else {"enable_thinking": False}
                )

                content = ""
                async for chunk in response:
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if delta and delta.content:
                        content += delta.content
                
                self.model_usage_count[current_model] += 1
                logger.info(f"模型 {current_model} 成功响应 - 长度: {len(content)}")

                if not content.strip():
                    return {
                        "success": False,
                        "content_indices": [],
                        "error": "LLM返回空内容"
                    }

                json_str = self.extract_json_from_response(content)
                
                if not json_str:
                    logger.error(f"无法从LLM响应中提取JSON {url}, 原始内容: {content[:200]}")
                    
                    if json_parse_retry < max_json_retries:
                        json_parse_retry += 1
                        logger.warning(f"JSON提取失败，重新提交任务 (第{json_parse_retry}/{max_json_retries}次)")
                        await asyncio.sleep(1)
                        continue
                    
                    return {
                        "success": False,
                        "content_indices": [],
                        "error": "无法提取JSON内容"
                    }

                try:
                    parsed = json.loads(json_str)
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON解析失败，尝试自动修复: {e}")
                    
                    fixed_json_str = self.fix_json_format(json_str)
                    
                    try:
                        parsed = json.loads(fixed_json_str)
                        logger.info("JSON自动修复成功")
                        
                    except json.JSONDecodeError as e2:
                        logger.error(f"JSON修复后仍然解析失败: {e2}")
                        
                        if json_parse_retry < max_json_retries:
                            json_parse_retry += 1
                            logger.warning(f"JSON解析失败，重新提交任务 (第{json_parse_retry}/{max_json_retries}次)")
                            await asyncio.sleep(1)
                            continue
                        
                        return {
                            "success": False,
                            "content_indices": [],
                            "error": f"JSON解析失败: {str(e2)}"
                        }
                
                is_success = parsed.get("status") == "success"
                content_indices = parsed.get("content_indices", [])
                

                if not content_indices:
                    # 没有content_indices，可能需要重试
                    if json_parse_retry < max_json_retries:
                        json_parse_retry += 1
                        logger.warning(f"LLM未返回content_indices，重新提交 (第{json_parse_retry}/{max_json_retries}次)")
                        await asyncio.sleep(1)
                        continue
                    else:
                        return {
                            "success": False,
                            "content_indices": [],
                            "error": "LLM未返回正文编号"
                        }

                if is_success:
                    return {
                        "success": True,
                        "content_indices": content_indices,
                        "error": ""
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
                        "success": False,
                        "content_indices": [],
                        "error": error_msg
                    }

            except Exception as e:
                error_str = str(e)
                
                if "429" in error_str or "rate_limit" in error_str.lower() or "too many requests" in error_str.lower():
                    logger.warning(f"模型 {current_model} 遇到速率限制 (429)")
                    
                    new_model = await self.handle_rate_limit_error(current_model)
                    rate_limit_cycle_count += 1
                    
                    if rate_limit_cycle_count > 0 and rate_limit_cycle_count % len(self.AVAILABLE_MODELS) == 0:
                        cycle_num = rate_limit_cycle_count // len(self.AVAILABLE_MODELS)
                        logger.warning(f"所有 {len(self.AVAILABLE_MODELS)} 个模型均遇到限制，第 {cycle_num} 轮轮换，等待后继续...")
                        await asyncio.sleep(2)
                    else:
                        await asyncio.sleep(0.5)
                    
                    continue
                else:
                    error_msg = f"LLM API调用失败 (模型: {current_model}): {error_str}"
                    logger.error(f"{error_msg} - URL: {url}")
                    
                    retry_count += 1
                    if retry_count < max_retries:
                        await asyncio.sleep(2)
                        continue
                    else:
                        return {
                            "success": False,
                            "content_indices": [],
                            "error": error_msg
                        }

        error_msg = f"LLM API调用失败，已重试 {max_retries} 次"
        logger.error(f"{error_msg} - URL: {url}")
        return {
            "success": False,
            "content_indices": [],
            "error": error_msg
        }


class ContentExtractor:
    """内容提取器 - 负责访问详情页并提取HTML内容"""
    
    def __init__(self, context: BrowserContext):
        self.context = context
    
    async def get_page_content(self, url: str, page: Page) -> str:
        """获取页面完整HTML内容"""
        max_retries = 10
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                if 'visit/link.do' in url:
                    await page.goto(url, wait_until='domcontentloaded', timeout=30000)

                actual_url = page.url
                if actual_url != url:
                    await page.goto(actual_url, wait_until='domcontentloaded', timeout=30000)

                html = await page.content()

                if len(html) < 200:
                    if retry_count < max_retries:
                        retry_count += 2
                        logger.warning(f"页面内容过短 (第{retry_count/2}次重试): {url}")
                        await asyncio.sleep(2)
                        continue
                    else:
                        return ""

                return html

            except asyncio.TimeoutError:
                if retry_count < max_retries:
                    retry_count += 1
                    logger.warning(f"页面访问超时 (第{retry_count}次重试): {url}")
                    await asyncio.sleep(3)
                    continue
                else:
                    return ""
            except Exception as e:
                if retry_count < max_retries:
                    retry_count += 1
                    logger.warning(f"获取页面内容失败 (第{retry_count}次重试) {url}: {e}")
                    await asyncio.sleep(3)
                    continue
                else:
                    return ""
        
        return ""
    
    def extract_numbered_chinese_blocks(self, html: str, max_blocks: int = 200) -> List[Tuple[int, str]]:
        """提取中文文本块"""
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(['script', 'style', 'noscript', 'link', 'iframe', 'svg', 'canvas', 'meta', 'head']):
            tag.decompose()
        
        def contains_chinese(text: str) -> bool:
            for char in text:
                if '\u4e00' <= char <= '\u9fff':
                    return True
            return False
        
        blocks: List[Tuple[int, str]] = []
        for text in soup.stripped_strings:
            parts = text.strip().split()
            candidate = ' '.join(parts)
            
            if len(candidate) <= 6:
                continue
            if not contains_chinese(candidate):
                continue
            blocks.append((len(blocks) + 1, candidate))
            if len(blocks) >= max_blocks:
                break
        
        return blocks
    
    async def extract_content(self, task: ContentTask) -> Tuple[SearchResult, Optional[List[Tuple[int, str]]]]:
        """执行内容提取任务 - 返回结果和blocks元组"""
        result = task.result
        
        if result.url in task.existing_urls:
            result.error = "URL已存在，跳过处理"
            logger.info(f"跳过已存在的URL: {result.url}")
            return result, None
        
        page = None
        try:
            if not result.url:
                result.error = "URL为空"
                return result, None

            page = await self.context.new_page()
            
            html = await self.get_page_content(result.url, page)
            
            if not html or len(html.strip()) < 500:
                result.error = "页面无法访问或内容为空"
                return result, None
            
            def contains_chinese(text: str) -> bool:
                for char in text:
                    if '\u4e00' <= char <= '\u9fff':
                        return True
                return False
            
            if not contains_chinese(html):
                result.error = "页面不包含中文内容"
                return result, None

            blocks = self.extract_numbered_chinese_blocks(html)
            if not blocks:
                result.error = "页面未提取到中文文本块"
                return result, None
            
            # 返回结果和blocks
            return result, blocks
            
        except Exception as e:
            result.error = f"处理页面失败: {str(e)}"
            return result, None
        finally:
            if page:
                await page.close()


class LLMAnalyzer:
    """LLM分析器 - 负责使用LLM分析文本块，仅用于正文提取"""
    
    def __init__(self, llm_client: LLMApiClient):
        self.llm_client = llm_client
    
    async def analyze_blocks(self, task: AnalysisTask) -> SearchResult:
        """执行LLM分析任务，仅提取正文"""
        result = task.result
        blocks = task.blocks
        
        if not blocks:
            result.error = "无可用文本块"
            return result
        
        max_llm_retries = 3
        llm_retry_count = 0
        
        while llm_retry_count <= max_llm_retries:
            llm_result = await self.llm_client.select_text_blocks(blocks, result.url)
            
            if not llm_result.get("success"):
                llm_error = llm_result.get("error") or "LLM分析失败"
                
                if llm_retry_count < max_llm_retries:
                    llm_retry_count += 1
                    logger.warning(f"LLM分析失败，重新提交任务 (第{llm_retry_count}/{max_llm_retries}次): {llm_error}")
                    await asyncio.sleep(1)
                    continue
                else:
                    result.error = llm_error
                    return result
            
            block_map = {idx: text for idx, text in blocks}
            
            # 只处理正文部分
            content_indices = self.normalize_indices(llm_result.get("content_indices"))
            
            if not content_indices:
                if llm_retry_count < max_llm_retries:
                    llm_retry_count += 1
                    logger.warning(f"LLM未返回有效正文编号，重新提交任务 (第{llm_retry_count}/{max_llm_retries}次)")
                    await asyncio.sleep(1)
                    continue
                else:
                    result.error = "LLM未返回正文编号"
                    return result
            
            content_segments = [block_map[idx] for idx in content_indices]
            result.content = "\n\n".join(content_segments).strip()
            
            if not result.content:
                result.error = "提取的正文为空"
            else:
                result.error = ""
            
            break
        
        return result

    def normalize_indices(self, raw_indices) -> List[int]:
        """规范化索引列表"""
        normalized = []
        for item in raw_indices or []:
            if isinstance(item, list) and len(item) == 2:
                try:
                    start, end = int(item[0]), int(item[1])
                    normalized.extend(range(start, end + 1))
                except (TypeError, ValueError):
                    continue
            else:
                try:
                    value = int(item)
                    normalized.append(value)
                except (TypeError, ValueError):
                    continue
        return sorted(set(normalized))


class ResultSaver:
    """结果保存器 - 负责保存结果到CSV"""
    
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.results_dir.mkdir(exist_ok=True)
    
    def sanitize_csv_content(self, content: str) -> str:
        """清理CSV内容中的特殊字符"""
        if not content:
            return content
            
        content = content.replace('\n', '\\n')
        content = content.replace('\r', '\\r')
        content = content.replace('\t', '\\t')
        content = content.replace('"', '\\"')
        content = content.replace("'", "\\'")
        
        cleaned = []
        for char in content:
            code = ord(char)
            if (0x00 <= code <= 0x08) or code == 0x0B or code == 0x0C or \
               (0x0E <= code <= 0x1F) or code == 0x7F:
                continue
            cleaned.append(char)
        
        content = ''.join(cleaned)
        content = content.strip()
        
        return content
    
    async def save_result(self, task: SaveTask):
        """执行保存任务"""
        city_folder = self.results_dir / task.city_name
        city_folder.mkdir(exist_ok=True)

        filename = city_folder / f"{task.city_name}-{task.keyword}-news.csv"

        data = {
            'keyword': self.sanitize_csv_content(task.result.keyword),
            'title': self.sanitize_csv_content(task.result.title),
            'url': self.sanitize_csv_content(task.result.url),
            'summary': self.sanitize_csv_content(task.result.summary),
            'date': self.sanitize_csv_content(task.result.date),
            'content': self.sanitize_csv_content(task.result.content),
            'column': self.sanitize_csv_content(task.result.column),
            'error': self.sanitize_csv_content(task.result.error),
            'page_num': task.result.page_num
        }

        df = pd.DataFrame([data])
        
        csv_params = {
            'index': False, 
            'encoding': 'utf-8',
            'quoting': 1,
            'escapechar': '\\',
            'doublequote': False
        }
        
        file_exists = filename.exists()
        
        if not file_exists:
            df.to_csv(filename, mode='w', header=True, **csv_params)
        else:
            df.to_csv(filename, mode='a', header=False, **csv_params)

        if task.result.error:
            logger.info(f"已保存结果到: {filename} (错误: {task.result.error[:50]}...)")
        else:
            logger.info(f"已保存结果到: {filename} (成功)")


class CrawlerOrchestrator:
    """爬虫编排器 - 协调各个组件，支持连续任务流"""
    
    def __init__(
        self,
        context: BrowserContext,
        llm_client: LLMApiClient,
        results_dir: Path,
        max_concurrent_search: int = 5,
        max_concurrent_content: int = 128,
        max_concurrent_analysis: int = 128,
        max_concurrent_save: int = 30
    ):
        self.search_fetcher = SearchResultFetcher(
            context, 
            "https://search.zj.gov.cn/jsearchfront/search.do"
        )
        self.content_extractor = ContentExtractor(context)
        self.llm_analyzer = LLMAnalyzer(llm_client)
        self.result_saver = ResultSaver(results_dir)
        
        # 各阶段的信号量
        self.search_sem = asyncio.Semaphore(max_concurrent_search)
        self.content_sem = asyncio.Semaphore(max_concurrent_content)
        self.analysis_sem = asyncio.Semaphore(max_concurrent_analysis)
        self.save_sem = asyncio.Semaphore(max_concurrent_save)
        
        # 任务队列
        self.search_queue: asyncio.Queue = asyncio.Queue()
        self.content_queue: asyncio.Queue = asyncio.Queue()
        self.analysis_queue: asyncio.Queue = asyncio.Queue()
        self.save_queue: asyncio.Queue = asyncio.Queue()
        
        # URL缓存
        self.url_cache: Dict[Tuple[str, str], Set[str]] = {}
        self.url_cache_lock = asyncio.Lock()
        
        # 工作器任务列表
        self.workers = []
        self.workers_started = False
    
    def load_existing_urls(self, city_name: str, keyword: str) -> Set[str]:
        """加载已存在的URL"""
        cache_key = (city_name, keyword)
        
        if cache_key in self.url_cache:
            return self.url_cache[cache_key]
        
        city_folder = self.result_saver.results_dir / city_name
        filename = city_folder / f"{city_name}-{keyword}-news.csv"
        
        if not filename.exists():
            self.url_cache[cache_key] = set()
            return self.url_cache[cache_key]
        
        try:
            df = pd.read_csv(filename, encoding='utf-8', usecols=['url'])
            existing_urls = set(df['url'].dropna().tolist())
            self.url_cache[cache_key] = existing_urls
            return existing_urls
        except Exception as e:
            logger.error(f"加载已有URL失败: {e}")
            self.url_cache[cache_key] = set()
            return self.url_cache[cache_key]
    
    async def search_worker(self):
        """搜索任务工作器"""
        while True:
            task = await self.search_queue.get()
            if task is None:  # 结束信号
                break
            
            async with self.search_sem:
                try:
                    results, total_pages = await self.search_fetcher.fetch_search_page(task)
                    
                    existing_urls = self.load_existing_urls(task.city_name, task.keyword)
                    
                    for result in results:
                        content_task = ContentTask(
                            result=result,
                            city_name=task.city_name,
                            keyword=task.keyword,
                            existing_urls=existing_urls
                        )
                        await self.content_queue.put(content_task)
                    
                    # 如果是第一页且有多页，添加后续页面任务
                    if task.page_num == 1 and total_pages > 1:
                        for page_num in range(2, total_pages + 1):
                            new_task = SearchTask(
                                city_name=task.city_name,
                                city_info=task.city_info,
                                keyword=task.keyword,
                                page_num=page_num
                            )
                            await self.search_queue.put(new_task)
                    
                except Exception as e:
                    logger.error(f"搜索任务失败: {e}")
                finally:
                    self.search_queue.task_done()
    
    async def content_worker(self):
        """内容提取任务工作器"""
        while True:
            task = await self.content_queue.get()
            if task is None:
                break
            
            async with self.content_sem:
                try:
                    result, blocks = await self.content_extractor.extract_content(task)
                    
                    if result.error == "URL已存在，跳过处理":
                        self.content_queue.task_done()
                        continue
                    
                    if result.error or not blocks:
                        # 直接保存错误结果
                        save_task = SaveTask(
                            result=result,
                            city_name=task.city_name,
                            keyword=task.keyword
                        )
                        await self.save_queue.put(save_task)
                    else:
                        # 发送到分析队列
                        analysis_task = AnalysisTask(
                            result=result,
                            blocks=blocks,
                            city_name=task.city_name,
                            keyword=task.keyword
                        )
                        await self.analysis_queue.put(analysis_task)
                    
                except Exception as e:
                    logger.error(f"内容提取任务失败: {e}")
                finally:
                    self.content_queue.task_done()
    
    async def analysis_worker(self):
        """LLM分析任务工作器"""
        while True:
            task = await self.analysis_queue.get()
            if task is None:
                break
            
            async with self.analysis_sem:
                try:
                    result = await self.llm_analyzer.analyze_blocks(task)
                    
                    save_task = SaveTask(
                        result=result,
                        city_name=task.city_name,
                        keyword=task.keyword
                    )
                    await self.save_queue.put(save_task)
                    
                except Exception as e:
                    logger.error(f"LLM分析任务失败: {e}")
                finally:
                    self.analysis_queue.task_done()
    
    async def save_worker(self):
        """保存任务工作器"""
        while True:
            task = await self.save_queue.get()
            if task is None:
                break
            
            async with self.save_sem:
                try:
                    await self.result_saver.save_result(task)
                    
                    # 更新URL缓存
                    if not task.result.error:
                        async with self.url_cache_lock:
                            cache_key = (task.city_name, task.keyword)
                            if cache_key not in self.url_cache:
                                self.url_cache[cache_key] = set()
                            self.url_cache[cache_key].add(task.result.url)
                    
                except Exception as e:
                    logger.error(f"保存任务失败: {e}")
                finally:
                    self.save_queue.task_done()
    
    async def start_workers(
        self,
        num_search_workers: int = 5,
        num_content_workers: int = 128,
        num_analysis_workers: int = 128,
        num_save_workers: int = 30
    ):
        """启动所有工作器（只调用一次）"""
        if self.workers_started:
            return
        
        self.workers_started = True
        
        # 搜索工作器
        for _ in range(num_search_workers):
            self.workers.append(asyncio.create_task(self.search_worker()))
        
        # 内容提取工作器
        for _ in range(num_content_workers):
            self.workers.append(asyncio.create_task(self.content_worker()))
        
        # LLM分析工作器
        for _ in range(num_analysis_workers):
            self.workers.append(asyncio.create_task(self.analysis_worker()))
        
        # 保存工作器
        for _ in range(num_save_workers):
            self.workers.append(asyncio.create_task(self.save_worker()))
        
        logger.info(f"已启动所有工作器: 搜索{num_search_workers}, 内容{num_content_workers}, 分析{num_analysis_workers}, 保存{num_save_workers}")
    
    async def add_city_keyword_task(self, city_name: str, city_info: Dict, keyword: str):
        """添加单个城市关键词任务（非阻塞）"""
        initial_task = SearchTask(
            city_name=city_name,
            city_info=city_info,
            keyword=keyword,
            page_num=1
        )
        await self.search_queue.put(initial_task)
        logger.info(f"已添加任务: {city_name} - {keyword}")
    
    async def wait_all_complete(self):
        """等待所有任务完成"""
        logger.info("等待所有搜索任务完成...")
        await self.search_queue.join()
        
        # 发送搜索结束信号
        for worker in self.workers:
            if not worker.done() and 'search_worker' in str(worker):
                await self.search_queue.put(None)
        
        logger.info("等待所有内容提取任务完成...")
        await self.content_queue.join()
        
        # 发送内容提取结束信号
        for worker in self.workers:
            if not worker.done() and 'content_worker' in str(worker):
                await self.content_queue.put(None)
        
        logger.info("等待所有分析任务完成...")
        await self.analysis_queue.join()
        
        # 发送分析结束信号
        for worker in self.workers:
            if not worker.done() and 'analysis_worker' in str(worker):
                await self.analysis_queue.put(None)
        
        logger.info("等待所有保存任务完成...")
        await self.save_queue.join()
        
        # 发送保存结束信号
        for worker in self.workers:
            if not worker.done() and 'save_worker' in str(worker):
                await self.save_queue.put(None)
        
        # 等待所有工作器结束
        await asyncio.gather(*self.workers, return_exceptions=True)
        logger.info("所有任务已完成")


class ZJCrawler:
    """浙江省政府网站爬虫 - 连续爬取版本"""

    def __init__(self, llm_api_key: str = ""):
        disable_system_proxies()
        self.llm_client = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        
        self.cache_dir = Path("browser_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
        self.progress_file = Path("crawl_progress.json")

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

    def load_target_cities(self, csv_file: str) -> List[str]:
        """从CSV文件加载313个目标城市列表"""
        try:
            df = pd.read_csv(csv_file)
            cities = df['city'].tolist() if 'city' in df.columns else df.iloc[:, 0].tolist()
            logger.info(f"加载了{len(cities)}个目标城市")
            return cities
        except Exception as e:
            logger.error(f"加载城市列表失败: {e}")
            return []

    def load_llm_api_key(self) -> str:
        """从文件中加载LLM API key，支持跨平台路径"""
        system_type = platform.system().lower()
        
        if system_type == "windows":
            api_key_file = Path.home() / ".siliconflow_apikey"
        else:
            api_key_file = Path.home() / ".siliconflow_apikey"
            
        try:
            with open(api_key_file, 'r', encoding='utf-8') as f:
                api_key = f.read().strip()
            if not api_key:
                raise ValueError("API key为空")
            logger.info(f"成功从文件加载API key: {api_key_file}")
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
        logger.info(f"✓ 完成: {source_city} -> {target_city} (时间: {timestamp})")
        self.save_progress(progress)

    async def run_crawler(self, cities_csv_file: str):
        """运行爬虫主程序 - 连续爬取版本"""
        logger.info("=" * 80)
        logger.info("启动连续爬取爬虫")
        logger.info("=" * 80)
        
        try:
            llm_api_key = self.load_llm_api_key()
        except Exception as e:
            logger.error(f"无法加载API key: {e}")
            return
        
        target_cities = self.load_target_cities(cities_csv_file)
        if not target_cities:
            logger.error("无法加载目标城市列表")
            return

        progress = self.load_progress()

        async with LLMApiClient(llm_api_key) as llm_client:
            self.llm_client = llm_client

            async with async_playwright() as p:
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
                        f'--disk-cache-dir={self.cache_dir}',
                        '--disk-cache-size=209715200',
                        '--disable-software-rasterizer',
                        '--disable-extensions',
                        '--disable-plugins',
                        '--disable-images',
                        '--blink-settings=imagesEnabled=false',
                        '--disable-javascript-harmony-shipping',
                    ],
                    proxy=None
                )
                
                self.context = await self.browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    viewport={"width": 1920, "height": 1080},
                    locale="zh-CN",
                    timezone_id="Asia/Shanghai",
                )

                orchestrator = CrawlerOrchestrator(
                    context=self.context,
                    llm_client=llm_client,
                    results_dir=self.results_dir,
                    max_concurrent_search=3,
                    max_concurrent_content=128,
                    max_concurrent_analysis=128,
                    max_concurrent_save=30
                )

                try:
                    # 一次性启动所有工作器
                    await orchestrator.start_workers(
                        num_search_workers=3,
                        num_content_workers=128,
                        num_analysis_workers=128,
                        num_save_workers=30
                    )
                    
                    # 统计信息
                    total_tasks = 0
                    skipped_tasks = 0
                    
                    # 连续添加所有城市对任务
                    for city_name, city_info in self.zj_cities.items():
                        logger.info(f"准备添加城市任务: {city_name}")

                        for target_city in target_cities:
                            target_city_name = str(target_city).strip()
                            if not target_city_name or target_city_name == city_name:
                                continue
                            
                            if self.is_city_pair_completed(progress, city_name, target_city_name):
                                skipped_tasks += 1
                                if skipped_tasks % 100 == 0:
                                    logger.info(f"已跳过 {skipped_tasks} 个已完成任务")
                                continue
                            
                            # 非阻塞地添加任务
                            await orchestrator.add_city_keyword_task(
                                city_name, city_info, target_city_name
                            )
                            total_tasks += 1
                            
                            # 每添加10个任务稍微延迟，避免队列瞬间爆满
                            if total_tasks % 10 == 0:
                                await asyncio.sleep(0.1)
                    
                    logger.info("=" * 80)
                    logger.info(f"所有任务已提交: 总计 {total_tasks} 个新任务, 跳过 {skipped_tasks} 个已完成任务")
                    logger.info("=" * 80)
                    
                    # 创建一个后台任务来监控队列状态
                    async def monitor_progress():
                        """监控爬取进度"""
                        last_report_time = time.time()
                        report_interval = 60  # 每60秒报告一次
                        
                        while True:
                            await asyncio.sleep(10)
                            
                            current_time = time.time()
                            if current_time - last_report_time >= report_interval:
                                search_size = orchestrator.search_queue.qsize()
                                content_size = orchestrator.content_queue.qsize()
                                analysis_size = orchestrator.analysis_queue.qsize()
                                save_size = orchestrator.save_queue.qsize()
                                
                                logger.info("=" * 80)
                                logger.info(f"队列状态: 搜索={search_size}, 内容={content_size}, 分析={analysis_size}, 保存={save_size}")
                                logger.info("=" * 80)
                                
                                last_report_time = current_time
                                
                                # 如果所有队列都为空，可能已经完成
                                if search_size == 0 and content_size == 0 and analysis_size == 0 and save_size == 0:
                                    logger.info("所有队列已清空，任务可能即将完成")
                                    break
                    
                    # 启动监控任务
                    monitor_task = asyncio.create_task(monitor_progress())
                    
                    # 等待所有任务完成
                    await orchestrator.wait_all_complete()
                    
                    # 取消监控任务
                    monitor_task.cancel()
                    try:
                        await monitor_task
                    except asyncio.CancelledError:
                        pass
                    
                    # 标记所有已处理的城市对为完成
                    logger.info("更新进度记录...")
                    for city_name in self.zj_cities.keys():
                        for target_city in target_cities:
                            target_city_name = str(target_city).strip()
                            if target_city_name and target_city_name != city_name:
                                self.mark_city_pair_completed(progress, city_name, target_city_name)
                    
                    logger.info("=" * 80)
                    logger.info("所有爬取任务已完成！")
                    logger.info("=" * 80)

                finally:
                    if self.context:
                        await self.context.close()
                    if self.browser:
                        await self.browser.close()


async def main():
    """主函数"""
    cities_csv_file = "target_cities.csv"
    crawler = ZJCrawler()
    await crawler.run_crawler(cities_csv_file)


if __name__ == "__main__":
    asyncio.run(main())