"""
FastAPI application for LangManus.
"""

import json
import logging
import os
from typing import Dict, List, Any, Optional, Union

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
import asyncio
from typing import AsyncGenerator, Dict, List, Any

from src.graph import build_graph
from src.config import TEAM_MEMBERS, BROWSER_HISTORY_DIR
from src.service.workflow_service import run_agent_workflow
from src.utils.log_handler import setup_logging, DEFAULT_LOG_DIR

# 设置日志系统
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="LangManus API",
    description="API for LangManus LangGraph-based agent workflow",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.on_event("startup")
async def startup_event():
    """在应用启动时初始化日志系统"""
    logger.info("初始化 LangManus API")
    setup_logging(debug=False, save_to_file=True)
    logger.info("日志系统已初始化")
    
    # 确保浏览器历史目录存在
    if not os.path.exists(BROWSER_HISTORY_DIR):
        os.makedirs(BROWSER_HISTORY_DIR)
        logger.info(f"创建浏览器历史目录: {BROWSER_HISTORY_DIR}")
        
    # 确保日志目录存在
    if not os.path.exists(DEFAULT_LOG_DIR):
        os.makedirs(DEFAULT_LOG_DIR)
        logger.info(f"创建日志目录: {DEFAULT_LOG_DIR}")

# Create the graph
graph = build_graph()


class ContentItem(BaseModel):
    type: str = Field(..., description="The type of content (text, image, etc.)")
    text: Optional[str] = Field(None, description="The text content if type is 'text'")
    image_url: Optional[str] = Field(
        None, description="The image URL if type is 'image'"
    )


class ChatMessage(BaseModel):
    role: str = Field(
        ..., description="The role of the message sender (user or assistant)"
    )
    content: Union[str, List[ContentItem]] = Field(
        ...,
        description="The content of the message, either a string or a list of content items",
    )


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="The conversation history")
    debug: Optional[bool] = Field(False, description="Whether to enable debug logging")
    deep_thinking_mode: Optional[bool] = Field(
        False, description="Whether to enable deep thinking mode"
    )
    search_before_planning: Optional[bool] = Field(
        False, description="Whether to search before planning"
    )


@app.post("/api/chat/stream")
async def chat_endpoint(request: ChatRequest, req: Request):
    """
    Chat endpoint for LangGraph invoke.

    Args:
        request: The chat request
        req: The FastAPI request object for connection state checking

    Returns:
        The streamed response
    """
    try:
        # Convert Pydantic models to dictionaries and normalize content format
        messages = []
        for msg in request.messages:
            message_dict = {"role": msg.role}

            # Handle both string content and list of content items
            if isinstance(msg.content, str):
                message_dict["content"] = msg.content
            else:
                # For content as a list, convert to the format expected by the workflow
                content_items = []
                for item in msg.content:
                    if item.type == "text" and item.text:
                        content_items.append({"type": "text", "text": item.text})
                    elif item.type == "image" and item.image_url:
                        content_items.append(
                            {"type": "image", "image_url": item.image_url}
                        )

                message_dict["content"] = content_items

            messages.append(message_dict)

        async def event_generator():
            try:
                async for event in run_agent_workflow(
                    messages,
                    request.debug,
                    request.deep_thinking_mode,
                    request.search_before_planning,
                ):
                    # Check if client is still connected
                    if await req.is_disconnected():
                        logger.info("Client disconnected, stopping workflow")
                        break
                    yield {
                        "event": event["event"],
                        "data": json.dumps(event["data"], ensure_ascii=False),
                    }
            except asyncio.CancelledError:
                logger.info("Stream processing cancelled")
                raise
            except Exception as e:
                logger.error(f"Error in workflow: {e}")
                raise

        return EventSourceResponse(
            event_generator(),
            media_type="text/event-stream",
            sep="\n",
        )
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/browser_history/{filename}")
async def get_browser_history_file(filename: str):
    """
    Get a specific browser history GIF file.

    Args:
        filename: The filename of the GIF to retrieve

    Returns:
        The GIF file
    """
    try:
        file_path = os.path.join(BROWSER_HISTORY_DIR, filename)
        if not os.path.exists(file_path) or not filename.endswith(".gif"):
            raise HTTPException(status_code=404, detail="File not found")

        return FileResponse(file_path, media_type="image/gif", filename=filename)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving browser history file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/logs")
async def list_log_files():
    """
    列出所有可用的日志文件。

    Returns:
        包含日志文件信息的JSON响应
    """
    try:
        logs_dir = DEFAULT_LOG_DIR
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
            return JSONResponse({"logs": []})
            
        log_files = []
        for file in os.listdir(logs_dir):
            if file.endswith(".log"):
                file_path = os.path.join(logs_dir, file)
                file_stat = os.stat(file_path)
                log_files.append({
                    "filename": file,
                    "size": file_stat.st_size,
                    "created": file_stat.st_ctime,
                    "modified": file_stat.st_mtime,
                    "is_debug": file.startswith("debug_")
                })
                
        # 按修改时间排序，最新的在前面
        log_files.sort(key=lambda x: x["modified"], reverse=True)
        
        return JSONResponse({"logs": log_files})
    except Exception as e:
        logger.error(f"Error listing log files: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
        
@app.get("/api/logs/{filename}")
async def get_log_file(filename: str, lines: Optional[int] = None, tail: Optional[bool] = False):
    """
    获取指定日志文件的内容。
    
    Args:
        filename: 日志文件名
        lines: 要返回的行数（如果指定）
        tail: 是否返回文件末尾的行（与lines一起使用）
        
    Returns:
        日志文件内容
    """
    try:
        file_path = os.path.join(DEFAULT_LOG_DIR, filename)
        if not os.path.exists(file_path) or not filename.endswith(".log"):
            raise HTTPException(status_code=404, detail="日志文件不存在")
        
        # 如果只需要部分内容
        if lines:
            with open(file_path, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()
                
            total_lines = len(all_lines)
            
            if tail:
                # 返回文件末尾的行
                start_line = max(0, total_lines - lines)
                result_lines = all_lines[start_line:]
            else:
                # 返回文件开头的行
                end_line = min(lines, total_lines)
                result_lines = all_lines[:end_line]
                
            return JSONResponse({
                "filename": filename,
                "content": "".join(result_lines),
                "total_lines": total_lines,
                "showing_lines": len(result_lines)
            })
        
        # 返回完整文件
        return FileResponse(
            file_path, 
            media_type="text/plain", 
            filename=filename
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving log file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/logs/{filename}")
async def delete_log_file(filename: str):
    """
    删除指定的日志文件。
    
    Args:
        filename: 日志文件名
        
    Returns:
        操作结果
    """
    try:
        file_path = os.path.join(DEFAULT_LOG_DIR, filename)
        if not os.path.exists(file_path) or not filename.endswith(".log"):
            raise HTTPException(status_code=404, detail="日志文件不存在")
            
        # 删除文件
        os.remove(file_path)
        logger.info(f"已删除日志文件: {filename}")
        
        return JSONResponse({"success": True, "message": f"日志文件 {filename} 已删除"})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除日志文件错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))
