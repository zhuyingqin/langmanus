import asyncio
import logging
import json
from pydantic import BaseModel, Field
from typing import Optional, ClassVar, Type
from langchain.tools import BaseTool
from browser_use import AgentHistoryList, Browser, BrowserConfig
from browser_use import Agent as BrowserAgent
from src.llms.llm import vl_llm
from src.tools.decorators import create_logged_tool
from src.config import (
    CHROME_INSTANCE_PATH,
    CHROME_HEADLESS,
    CHROME_PROXY_SERVER,
    CHROME_PROXY_USERNAME,
    CHROME_PROXY_PASSWORD,
    BROWSER_HISTORY_DIR,
)
import uuid

# Configure logging
logger = logging.getLogger(__name__)

browser_config = BrowserConfig(
    headless=CHROME_HEADLESS,
    chrome_instance_path=CHROME_INSTANCE_PATH,
)
if CHROME_PROXY_SERVER:
    proxy_config = {
        "server": CHROME_PROXY_SERVER,
    }
    if CHROME_PROXY_USERNAME:
        proxy_config["username"] = CHROME_PROXY_USERNAME
    if CHROME_PROXY_PASSWORD:
        proxy_config["password"] = CHROME_PROXY_PASSWORD
    browser_config.proxy = proxy_config

expected_browser = Browser(config=browser_config)


class BrowserUseInput(BaseModel):
    """Input for WriteFileTool."""

    instruction: str = Field(..., description="The instruction to use browser")


class BrowserTool(BaseTool):
    name: ClassVar[str] = "browser"
    args_schema: Type[BaseModel] = BrowserUseInput
    description: ClassVar[str] = (
        "Use this tool to interact with web browsers. Input should be a natural language description of what you want to do with the browser, such as 'Go to google.com and search for browser-use', or 'Navigate to Reddit and find the top post about AI'."
    )

    _agent: Optional[BrowserAgent] = None

    def _generate_browser_result(
        self, result_content: str, generated_gif_path: str
    ) -> dict:
        # 确保结果内容不为空
        if result_content is None or result_content == "":
            result_content = "浏览器任务执行完成，但未返回有效内容。请检查任务是否成功或提供更明确的指令。"
            
        return {
            "result_content": result_content,
            "generated_gif_path": generated_gif_path,
        }

    def _run(self, instruction: str) -> str:
        """Run the browser task synchronously."""
        generated_gif_path = f"{BROWSER_HISTORY_DIR}/{uuid.uuid4()}.gif"
        
        # 记录开始执行浏览器任务
        logger.info(f"开始执行浏览器任务: {instruction}")
        
        self._agent = BrowserAgent(
            task=instruction,  # Will be set per request
            llm=vl_llm,
            browser=expected_browser,
            generate_gif=generated_gif_path,
        )

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self._agent.run())

                # 检查结果并生成有效的响应
                if isinstance(result, AgentHistoryList):
                    final_result = result.final_result()
                    # 确保最终结果不为空
                    if final_result is None or final_result == "":
                        final_result = "浏览器任务执行完成，但未返回结果。GIF截图可能提供了更多信息。"
                    
                    return json.dumps(
                        self._generate_browser_result(
                            final_result, generated_gif_path
                        )
                    )
                else:
                    # 确保结果不为空
                    if result is None or result == "":
                        result = "浏览器任务执行完成，但未返回文本结果。GIF截图可能提供了更多信息。"
                        
                    return json.dumps(
                        self._generate_browser_result(result, generated_gif_path)
                    )
            finally:
                loop.close()
        except Exception as e:
            error_message = f"执行浏览器任务时出错: {str(e)}"
            logger.error(error_message)
            # 返回错误信息和GIF路径，确保返回值有效
            return json.dumps(
                self._generate_browser_result(error_message, generated_gif_path)
            )

    async def terminate(self):
        """Terminate the browser agent if it exists."""
        if self._agent and self._agent.browser:
            try:
                await self._agent.browser.close()
            except Exception as e:
                logger.error(f"Error terminating browser agent: {str(e)}")
        self._agent = None

    async def _arun(self, instruction: str) -> str:
        """Run the browser task asynchronously."""
        generated_gif_path = f"{BROWSER_HISTORY_DIR}/{uuid.uuid4()}.gif"
        
        # 记录开始执行浏览器任务
        logger.info(f"开始异步执行浏览器任务: {instruction}")
        
        self._agent = BrowserAgent(
            task=instruction,
            llm=vl_llm,
            browser=expected_browser,
            generate_gif=generated_gif_path,  # Will be set per request
        )
        try:
            result = await self._agent.run()
            
            # 检查结果并生成有效的响应
            if isinstance(result, AgentHistoryList):
                final_result = result.final_result()
                # 确保最终结果不为空
                if final_result is None or final_result == "":
                    final_result = "浏览器任务执行完成，但未返回结果。GIF截图可能提供了更多信息。"
                
                return json.dumps(
                    self._generate_browser_result(
                        final_result, generated_gif_path
                    )
                )
            else:
                # 确保结果不为空
                if result is None or result == "":
                    result = "浏览器任务执行完成，但未返回文本结果。GIF截图可能提供了更多信息。"
                    
                return json.dumps(
                    self._generate_browser_result(result, generated_gif_path)
                )
        except Exception as e:
            error_message = f"异步执行浏览器任务时出错: {str(e)}"
            logger.error(error_message)
            # 返回错误信息和GIF路径，确保返回值有效
            return json.dumps(
                self._generate_browser_result(error_message, generated_gif_path)
            )
        finally:
            await self.terminate()


BrowserTool = create_logged_tool(BrowserTool)
browser_tool = BrowserTool()
