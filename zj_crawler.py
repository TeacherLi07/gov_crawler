import asyncio
import csv
import json
import os
import re
import time
import platform
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from urllib.parse import urljoin, urlparse
import pandas as pd
from playwright.async_api import async_playwright, Page, Browser, BrowserContext
import aiohttp
import logging
from dataclasses import dataclass
from openai import AsyncOpenAI
from bs4 import BeautifulSoup

DEBUG = 1

logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crawler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def debug_log(message: str):
    if DEBUG:
        logger.debug(message)


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


class LLMApiClient:
    """LLM API客户端，使用OpenAI SDK调用SiliconFlow服务"""

    def __init__(self, api_key: str, base_url: str = "https://api.siliconflow.cn/v1"):
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url
        )

    async def __aenter__(self):
        debug_log("LLMApiClient进入上下文管理器")
        return self

    async def __aexit__(self, exc_type, exc, tb):
        debug_log("LLMApiClient退出上下文管理器")
        return False

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

    async def extract_content(self, html: str, url: str) -> Dict[str, str]:
        """使用LLM提取页面标题和正文"""
        # 优化：只推送主要含中文正文的内容
        main_content = self.extract_main_html_content(html)
        prompt = f"""
        请从以下HTML内容中提取页面的标题和正文内容。只需要纯文本，忽略图片、视频、表格等。

        要求：
        1. title: 页面的主标题
        2. content: 页面的正文内容，保持段落结构，去除导航、广告等无关内容
        4. status: 成功时为"success"，失败或不确定时为"error"
        5. message: 当status为"error"时给出简要说明，成功时可留空

        请严格返回以下JSON格式，不要包含其他解释文字：
        {{"status": "success 或 error", "title": "页面标题", "content": "正文内容", "message": "补充说明"}}

        HTML内容：
        {main_content[:100000]}...

        URL: {url}
        """
        debug_log(f"发送到LLM的URL: {url}")
        debug_log(f"LLM提示词前200字符: {prompt[:200]}")

        try:
            response = await self.client.chat.completions.create(
                model="Qwen/Qwen3-Next-80B-A3B-Instruct",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant designed to output JSON. 返回格式必须严格遵循用户提供的JSON结构。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=192000,
                temperature=0.1,
                frequency_penalty=0.05,
                stream=True,
                timeout=60
            )

            content = ""
            async for chunk in response:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    fragment = delta.content
                    content += fragment
                    # debug_log(f"收到LLM片段长度: {len(fragment)}")
            debug_log(f"LLM完整响应长度: {len(content)}")

            if not content.strip():
                debug_log("LLM返回空内容")
                return {
                    "title": "",
                    "content": "",
                    "status": "error",
                    "message": "",
                    "error": "LLM返回空内容"
                }

            try:
                parsed = json.loads(content.strip())
                status = (parsed.get("status") or "").lower()
                message = parsed.get("message", "")
                debug_log(f"LLM解析状态: {status}, message: {message}")
                return {
                    "title": parsed.get("title", ""),
                    "content": parsed.get("content", ""),
                    "status": status,
                    "message": message,
                    "error": "" if status == "success" else (message or "LLM返回非成功状态")
                }
            except json.JSONDecodeError as e:
                logger.error(f"JSON解析失败 {url}: {e}, 原始内容: {content[:200]}")
                return {
                    "title": "",
                    "content": content.strip(),
                    "status": "error",
                    "message": "",
                    "error": f"JSON解析失败: {str(e)}"
                }

        except asyncio.TimeoutError:
            error_msg = f"LLM API调用超时: {url}"
            logger.error(error_msg)
            return {"title": "", "content": "", "status": "error", "message": "", "error": error_msg}

        except Exception as e:
            error_msg = f"LLM API调用失败: {str(e)}"
            logger.error(f"{error_msg} - URL: {url}")
            return {"title": "", "content": "", "status": "error", "message": "", "error": error_msg}

    async def extract_element_paths(self, html: str, url: str) -> Dict[str, any]:
        """使用LLM分析HTML结构并返回元素路径"""
        # 优化：只推送主要含中文正文的内容
        main_content = self.extract_main_html_content(html)
        prompt = f"""
        请分析以下HTML内容的结构，找出页面标题和正文内容所在的CSS选择器路径。

        要求：
        1. title_selectors: 包含页面主标题的CSS选择器数组，可能有多个候选
        2. content_selectors: 包含正文内容的CSS选择器数组，这些元素可能需要拼接组成完整正文
        3. status: 成功时为"success"，失败或不确定时为"error"
        4. message: 当status为"error"时给出简要说明，成功时可留空

        注意：
        - 标题通常在 h1, h2, .title, .article-title 等元素中
        - 正文可能分布在多个段落 p, div, .content, .article-content 等元素中
        - 所有content_selectors匹配的元素都会被提取并按顺序拼接
        - CSS选择器要准确定位到实际内容，避免包含导航等无关内容
        
        请严格返回以下JSON格式：
        {{
            "status": "success 或 error",
            "title_selectors": ["h1", ".page-title", ".article-title"],
            "content_selectors": [".article-content p", ".content-body", "article div"],
            "message": "补充说明"
        }}

        HTML内容：
        {main_content[:50000]}...

        URL: {url}
        """
        debug_log(f"发送到LLM的URL: {url}")
        debug_log(f"LLM提示词前200字符: {prompt[:200]}")

        try:
            response = await self.client.chat.completions.create(
                model="Qwen/Qwen3-Next-80B-A3B-Instruct",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant designed to output JSON. 你需要分析HTML结构并返回CSS选择器路径，内容选择器用于拼接多个元素组成正文。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4000,
                temperature=0.1,
                frequency_penalty=0.05,
                stream=True,
                timeout=30
            )

            content = ""
            async for chunk in response:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    fragment = delta.content
                    content += fragment
                    debug_log(f"收到LLM片段长度: {len(fragment)}")
            debug_log(f"LLM完整响应长度: {len(content)}")

            if not content.strip():
                debug_log("LLM返回空内容")
                return {
                    "status": "error",
                    "title_selectors": [],
                    "content_selectors": [],
                    "message": "LLM返回空内容",
                    "error": "LLM返回空内容"
                }

            try:
                parsed = json.loads(content.strip())
                status = (parsed.get("status") or "").lower()
                message = parsed.get("message", "")
                debug_log(f"LLM解析状态: {status}, message: {message}")
                
                return {
                    "status": status,
                    "title_selectors": parsed.get("title_selectors", []),
                    "content_selectors": parsed.get("content_selectors", []),
                    "message": message,
                    "error": "" if status == "success" else (message or "LLM返回非成功状态")
                }
            except json.JSONDecodeError as e:
                logger.error(f"JSON解析失败 {url}: {e}, 原始内容: {content[:200]}")
                return {
                    "status": "error",
                    "title_selectors": [],
                    "content_selectors": [],
                    "message": f"JSON解析失败: {str(e)}",
                    "error": f"JSON解析失败: {str(e)}"
                }

        except asyncio.TimeoutError:
            error_msg = f"LLM API调用超时: {url}"
            logger.error(error_msg)
            return {
                "status": "error", "title_selectors": [], "content_selectors": [], 
                "message": "", "error": error_msg
            }

        except Exception as e:
            error_msg = f"LLM API调用失败: {str(e)}"
            logger.error(f"{error_msg} - URL: {url}")
            return {
                "status": "error", "title_selectors": [], "content_selectors": [], 
                "message": "", "error": error_msg
            }


class ZJCrawler:
    """浙江省政府网站爬虫"""

    def __init__(self, llm_api_key: str, max_concurrent_pages: int = 5):
        debug_log("初始化ZJCrawler")
        self.llm_client = LLMApiClient(llm_api_key)
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.max_concurrent_pages = max_concurrent_pages

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
                        await page.screenshot(path=str(screenshot_f ilename), full_page=True)
                        debug_log(f"已保存调试截图: {screenshot_filename}")
                        logger.warning(f"搜索页面无结果，已保存截图到: {screenshot_filename}")
                    except Exception as screenshot_error:
                        debug_log(f"保存调试截图失败: {screenshot_error}")
                
                # 检查是否包含无结果提示
                # if "很抱歉！未能搜到您想要的内容" in page_html:
                #     debug_log("页面显示无搜索结果")
                #     return results, total_pages
                    
                debug_log(f"失败页面HTML关键部分: {page_html[page_html.find('<body'):page_html.find('</body>') + 7] if '<body' in page_html else page_html[:1000]}")
            except Exception as inner_exc:
                debug_log(f"获取失败页面HTML时出错: {inner_exc}")
            logger.error(f"提取搜索结果失败: {e}")

        return results, total_pages

    async def get_page_content(self, url: str, page: Page) -> str:
        """获取页面完整HTML内容，支持重试机制"""
        start_time = time.perf_counter()
        debug_log(f"开始获取页面内容: {url}")
        
        max_retries = 2
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                if 'visit/link.do' in url:
                    debug_log(f"即将访问搜索页面 (第{retry_count + 1}次尝试): {url}")
                    await page.goto(url, wait_until='domcontentloaded', timeout=30000)
                    debug_log(f"搜索页面实际URL: {page.url}")
                    await page.wait_for_timeout(3000)
                    debug_log("搜索页面初步加载完成")

                actual_url = page.url
                if actual_url != url:
                    debug_log(f"检测到跳转至: {actual_url}")
                    await page.goto(actual_url, wait_until='domcontentloaded', timeout=30000)

                await page.wait_for_timeout(3000)
                html = await page.content()
                duration = time.perf_counter() - start_time
                debug_log(f"获取页面耗时: {duration:.2f}s, HTML长度: {len(html)}")

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
        city_folder = Path(city_name)
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

    def extract_content_by_selectors(self, html: str, title_selectors: List[str], 
                                   content_selectors: List[str]) -> Dict[str, str]:
        """根据CSS选择器提取页面内容，支持多个元素拼接"""
        soup = BeautifulSoup(html, "lxml")
        
        # 提取标题 - 尝试所有候选选择器
        title = ""
        for selector in title_selectors:
            try:
                elements = soup.select(selector)
                if elements:
                    title = elements[0].get_text(strip=True)
                    debug_log(f"通过选择器 {selector} 提取到标题: {title[:50]}...")
                    break
            except Exception as e:
                debug_log(f"标题选择器失败 {selector}: {e}")
                continue
        
        # 提取内容 - 拼接所有匹配的元素
        content_parts = []
        for selector in content_selectors:
            try:
                elements = soup.select(selector)
                debug_log(f"选择器 {selector} 匹配到 {len(elements)} 个元素")
                
                for elem in elements:
                    # 获取纯文本，保持段落结构
                    text = elem.get_text(separator='\n', strip=True)
                    if text and len(text) > 20:  # 过滤太短的内容片段
                        content_parts.append(text)
                        debug_log(f"通过选择器 {selector} 提取内容片段: {len(text)}字符")
                        
            except Exception as e:
                debug_log(f"内容选择器失败 {selector}: {e}")
                continue
        
        # 拼接所有内容片段
        content = '\n\n'.join(content_parts) if content_parts else ""
        
        debug_log(f"共提取到 {len(content_parts)} 个内容片段，总长度: {len(content)}字符")
        
        return {
            "title": title,
            "content": content,
            "extracted_by": "css_selectors",
            "content_parts_count": len(content_parts)
        }

    async def process_single_result(self, result: SearchResult, semaphore: asyncio.Semaphore) -> SearchResult:
        """处理单个搜索结果，获取内容并使用选择器提取"""
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
                
                # 预检查HTML中是否包含中文内容
                chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
                if not chinese_pattern.search(html):
                    result.error = "页面不包含中文内容"
                    debug_log(f"跳过处理，页面无中文内容: {result.url}")
                    return result

                # HTML有效，使用LLM获取选择器路径
                debug_log(f"开始LLM分析页面结构: {result.url}")
                selector_result = await self.llm_client.extract_element_paths(html, result.url)
                
                if selector_result.get("status") != "success" or selector_result.get("error"):
                    result.error = f"LLM分析失败: {selector_result.get('error', '未知错误')}"
                    debug_log(f"LLM分析失败: {result.url}")
                    return result

                # 使用选择器提取内容
                title_selectors = selector_result.get("title_selectors", [])
                content_selectors = selector_result.get("content_selectors", [])
                
                debug_log(f"使用选择器提取内容: {result.url}")
                debug_log(f"标题选择器: {title_selectors}")
                debug_log(f"内容选择器: {content_selectors}")
                
                extracted = self.extract_content_by_selectors(html, title_selectors, content_selectors)
                
                result.content = extracted.get("content", "")
                
                # 如果原标题为空，使用提取的标题
                if not result.title.strip() and extracted.get("title"):
                    result.title = extracted.get("title")
                
                # 检查提取结果质量
                if not result.content.strip():
                    result.error = "选择器未能提取到有效内容"
                elif len(result.content) < 100:
                    result.error = "提取的内容过短，可能不完整"
                else:
                    debug_log(f"成功提取内容: 标题={bool(result.title)}, 正文长度={len(result.content)}, 片段数={extracted.get('content_parts_count', 0)}")

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

    async def process_results_individually(self, city_name: str, keyword: str, results: List[SearchResult], page_num: int) -> List[SearchResult]:
        """逐个处理搜索结果并立即保存"""
        if not results:
            return results

        debug_log(f"开始逐个处理 {len(results)} 个搜索结果")
        
        # 创建信号量来限制并发数量
        semaphore = asyncio.Semaphore(self.max_concurrent_pages)
        processed_results = []
        
        # 创建任务列表但不立即执行全部
        tasks = []
        for result in results:
            task = asyncio.create_task(self.process_single_result(result, semaphore))
            tasks.append(task)
        
        # 逐个等待任务完成并立即保存
        for i, task in enumerate(tasks, 1):
            try:
                processed_result = await task
                processed_results.append(processed_result)
                
                # 立即保存单个结果
                await self.save_single_result_to_csv(city_name, keyword, processed_result, page_num)
                
                debug_log(f"已处理并保存第{i}/{len(results)}个结果")
                
            except Exception as e:
                logger.error(f"处理第{i}个结果时出现异常: {e}")
                # 创建错误结果
                error_result = results[i-1] if i <= len(results) else SearchResult("", "", "", "", "")
                error_result.error = f"处理异常: {str(e)}"
                processed_results.append(error_result)
                
                # 保存错误结果
                await self.save_single_result_to_csv(city_name, keyword, error_result, page_num)

        success_count = len([r for r in processed_results if not r.error])
        error_count = len(processed_results) - success_count
        debug_log(f"逐个处理完成: 成功{success_count}个, 失败{error_count}个")
        
        return processed_results

    async def process_single_city_keyword(self, city_name: str, city_info: Dict,
                                          keyword: str, session: aiohttp.ClientSession) -> List[SearchResult]:
        """处理单个城市的单个关键词搜索"""
        all_results = []
        page_num = 1
        max_pages = None

        debug_log(f"开始处理城市 {city_name} 的关键词 {keyword}")
        search_page = await self.context.new_page()

        try:
            while True:
                url = self.build_search_url(city_info, keyword, page_num)
                logger.info(f"正在爬取: {city_name} - {keyword} - 第{page_num}页")

                results = []
                retry_count = 0
                max_retries = 3
                
                # 获取搜索结果页面（保持原有逻辑）
                while retry_count < max_retries:
                    try:
                        debug_log(f"尝试访问页面 (第{retry_count + 1}次): {url}")
                        
                        # 设置额外的请求头
                        await search_page.set_extra_http_headers({
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
                        
                        await search_page.goto(url, wait_until='networkidle', timeout=30000)
                        debug_log(f"页面加载完成，当前URL: {search_page.url}")
                        
                        # 额外等待确保页面完全渲染
                        await search_page.wait_for_timeout(3000)
                        
                        # 检查页面是否正常加载
                        page_title = await search_page.title()
                        debug_log(f"页面标题: {page_title}")
                        
                        # 尝试滚动页面以触发懒加载
                        await search_page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                        await search_page.wait_for_timeout(1000)
                        await search_page.evaluate("window.scrollTo(0, 0)")
                        await search_page.wait_for_timeout(1000)

                        results, total_pages = await self.extract_search_results(search_page)
                        
                        if not results and retry_count < max_retries - 1:
                            debug_log(f"第{retry_count + 1}次尝试无结果，{2}秒后重试")
                            await asyncio.sleep(2)
                            retry_count += 1
                            continue
                        else:
                            break

                    except Exception as e:
                        retry_count += 1
                        error_msg = f"访问页面失败 (第{retry_count}次): {str(e)}"
                        logger.warning(error_msg)
                        
                        if retry_count >= max_retries:
                            logger.error(f"重试{max_retries}次后仍然失败: {url}")
                            break
                        else:
                            await asyncio.sleep(3)

                if not results:
                    logger.warning(f"第{page_num}页无搜索结果，结束该关键词搜索")
                    break

                # 设置关键词
                for result in results:
                    result.keyword = keyword

                # **逐个处理搜索结果内容提取并立即保存**
                logger.info(f"开始逐个处理第{page_num}页的{len(results)}个搜索结果")
                processed_results = await self.process_results_individually(city_name, keyword, results, page_num)
                
                all_results.extend(processed_results)
                debug_log(f"第{page_num}页已处理完成，累计结果数量: {len(all_results)}")

                if max_pages is None and total_pages > 0:
                    max_pages = total_pages
                    logger.info(f"{city_name} - {keyword}: 共{max_pages}页")

                if max_pages and page_num >= max_pages:
                    debug_log("达到最大页数，结束分页")
                    break

                page_num += 1
                await asyncio.sleep(1)

        finally:
            await search_page.close()

        logger.info(f"完成 {city_name} - {keyword}: 总共{len(all_results)}条结果，已逐个保存")
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
        
        # 移除其他可能的控制字符
        content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', content)
        
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

    async def run_crawler(self, cities_csv_file: str):
        """运行爬虫主程序"""
        logger.info("启动爬虫")
        
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
                        '--disable-features=VizDisplayCompositor'
                    ]
                )
                debug_log("Chromium浏览器已启动")
                
                # 增加浏览器上下文的并发连接限制
                self.context = await self.browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                    viewport={"width": 1920, "height": 1080},
                    locale="zh-CN",
                    timezone_id="Asia/Shanghai",
                    java_script_enabled=True,
                    ignore_https_errors=True,
                    extra_http_headers={
                        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                        "Upgrade-Insecure-Requests": "1",
                        "DNT": "1"
                    }
                )
                debug_log(f"浏览器上下文已创建，最大并发页面数: {self.max_concurrent_pages}")

                try:
                    for city_name, city_info in self.zj_cities.items():
                        logger.info(f"开始处理城市: {city_name}")

                        async with aiohttp.ClientSession() as session:
                            for target_city in target_cities:
                                target_city_name = str(target_city).strip()
                                if not target_city_name:
                                    debug_log(f"跳过空白城市记录: {target_city}")
                                    continue
                                if target_city_name == city_name:
                                    logger.info(f"跳过自身城市: {city_name}")
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

                                    await asyncio.sleep(3)  # 增加延时

                                except Exception as e:
                                    logger.error(f"处理 {city_name}-{target_city_name} 失败: {e}")
                                    continue

                        logger.info(f"完成城市: {city_name}")

                finally:
                    if self.context:
                        await self.context.close()
                        self.context = None
                        debug_log("Chromium浏览器上下文已关闭")
                    if self.browser:
                        await self.browser.close()
                        debug_log("Chromium浏览器已关闭")


async def main():
    """主函数"""
    cities_csv_file = "target_cities.csv"

    # 不再需要手动提供API key，直接从文件读取
    crawler = ZJCrawler("", max_concurrent_pages=5)  # API key会从文件自动加载
    await crawler.run_crawler(cities_csv_file)


if __name__ == "__main__":
    asyncio.run(main())