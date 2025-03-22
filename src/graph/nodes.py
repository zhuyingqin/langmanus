import logging
import json
import json_repair
import logging
from copy import deepcopy
from typing import Literal
from langchain_core.messages import HumanMessage, BaseMessage

import json_repair
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from src.agents import research_agent, coder_agent, browser_agent
from src.llms.llm import get_llm_by_type
from src.config import TEAM_MEMBERS
from src.config.agents import AGENT_LLM_MAP
from src.prompts.template import apply_prompt_template
from src.tools.search import tavily_tool
from src.utils.json_utils import repair_json_output
from .types import State, Router

logger = logging.getLogger(__name__)

RESPONSE_FORMAT = "Response from {}:\n\n<response>\n{}\n</response>\n\n*Please execute the next step.*"


def research_node(state: State) -> Command[Literal["supervisor"]]:
    """Node for the researcher agent that performs research tasks."""
    logger.info("Research agent starting task")
    
    try:
        # 验证输入状态
        if not state.get("messages"):
            logger.warning("No messages in state for research agent, ending early")
            return Command(
                update={
                    "messages": [
                        HumanMessage(
                            content="无法执行研究任务：输入消息为空",
                            name="researcher",
                        )
                    ]
                },
                goto="supervisor",
            )
            
        # 调用研究代理
        result = research_agent.invoke(state)
        logger.info("Research agent completed task")
        
        # 验证返回结果
        if not result or "messages" not in result or not result["messages"]:
            logger.error("Research agent returned empty result")
            return Command(
                update={
                    "messages": [
                        HumanMessage(
                            content="研究任务未返回有效结果",
                            name="researcher",
                        )
                    ]
                },
                goto="supervisor",
            )
            
        # 获取最后一条消息内容
        last_message = result["messages"][-1]
        if not hasattr(last_message, "content") or last_message.content is None:
            logger.error("Last message from research agent has null content")
            return Command(
                update={
                    "messages": [
                        HumanMessage(
                            content="研究任务返回了空内容",
                            name="researcher",
                        )
                    ]
                },
                goto="supervisor",
            )
            
        response_content = last_message.content
        # 尝试修复可能的JSON输出
        response_content = repair_json_output(response_content)
        logger.debug(f"Research agent response: {response_content}")
        
        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content=response_content,
                        name="researcher",
                    )
                ]
            },
            goto="supervisor",
        )
    except Exception as e:
        logger.error(f"Error in research node: {str(e)}")
        # 发生错误时提供友好的错误消息
        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content=f"执行研究任务时发生错误: {str(e)}",
                        name="researcher",
                    )
                ]
            },
            goto="supervisor",
        )


def code_node(state: State) -> Command[Literal["supervisor"]]:
    """Node for the coder agent that executes Python code."""
    logger.info("Code agent starting task")
    
    try:
        # 验证输入状态
        if not state.get("messages"):
            logger.warning("No messages in state for code agent, ending early")
            return Command(
                update={
                    "messages": [
                        HumanMessage(
                            content="无法执行代码任务：输入消息为空",
                            name="coder",
                        )
                    ]
                },
                goto="supervisor",
            )
            
        # 调用代码代理
        result = coder_agent.invoke(state)
        logger.info("Code agent completed task")
        
        # 验证返回结果
        if not result or "messages" not in result or not result["messages"]:
            logger.error("Code agent returned empty result")
            return Command(
                update={
                    "messages": [
                        HumanMessage(
                            content="代码任务未返回有效结果",
                            name="coder",
                        )
                    ]
                },
                goto="supervisor",
            )
            
        # 获取最后一条消息内容
        last_message = result["messages"][-1]
        if not hasattr(last_message, "content") or last_message.content is None:
            logger.error("Last message from code agent has null content")
            return Command(
                update={
                    "messages": [
                        HumanMessage(
                            content="代码任务返回了空内容",
                            name="coder",
                        )
                    ]
                },
                goto="supervisor",
            )
            
        response_content = last_message.content
        # 尝试修复可能的JSON输出
        response_content = repair_json_output(response_content)
        logger.debug(f"Code agent response: {response_content}")
        
        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content=response_content,
                        name="coder",
                    )
                ]
            },
            goto="supervisor",
        )
    except Exception as e:
        logger.error(f"Error in code node: {str(e)}")
        # 发生错误时提供友好的错误消息
        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content=f"执行代码任务时发生错误: {str(e)}",
                        name="coder",
                    )
                ]
            },
            goto="supervisor",
        )


def browser_node(state: State) -> Command[Literal["supervisor"]]:
    """Node for the browser agent that performs web browsing tasks."""
    logger.info("Browser agent starting task")
    
    try:
        # 验证输入状态
        if not state.get("messages"):
            logger.warning("No messages in state for browser agent, ending early")
            return Command(
                update={
                    "messages": [
                        HumanMessage(
                            content="无法执行浏览器任务：输入消息为空",
                            name="browser",
                        )
                    ]
                },
                goto="supervisor",
            )
            
        # 调用浏览器代理
        result = browser_agent.invoke(state)
        logger.info("Browser agent completed task")
        
        # 验证返回结果
        if not result or "messages" not in result or not result["messages"]:
            logger.error("Browser agent returned empty result")
            return Command(
                update={
                    "messages": [
                        HumanMessage(
                            content="浏览器任务未返回有效结果",
                            name="browser",
                        )
                    ]
                },
                goto="supervisor",
            )
            
        # 获取最后一条消息内容
        last_message = result["messages"][-1]
        if not hasattr(last_message, "content") or last_message.content is None:
            logger.error("Last message from browser agent has null content")
            return Command(
                update={
                    "messages": [
                        HumanMessage(
                            content="浏览器任务返回了空内容",
                            name="browser",
                        )
                    ]
                },
                goto="supervisor",
            )
            
        response_content = last_message.content
        # 尝试修复可能的JSON输出
        response_content = repair_json_output(response_content)
        logger.debug(f"Browser agent response: {response_content}")
        
        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content=response_content,
                        name="browser",
                    )
                ]
            },
            goto="supervisor",
        )
    except Exception as e:
        logger.error(f"Error in browser node: {str(e)}")
        # 发生错误时提供友好的错误消息
        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content=f"执行浏览器任务时发生错误: {str(e)}",
                        name="browser",
                    )
                ]
            },
            goto="supervisor",
        )


def reflection_node(state: State) -> Command[Literal["supervisor", "reporter", "__end__"]]:
    """反思节点，批评和改进当前解决方案。
    
    该节点分析当前状态，包括所有消息和迄今为止的决策，
    以识别当前方法的潜在改进或问题。
    """
    logger.info("Reflection agent starting analysis")
    
    try:
        # 确保state中有消息
        if not state.get("messages"):
            logger.warning("No messages found in state, skipping reflection")
            return Command(
                update={
                    "messages": [
                        HumanMessage(
                            content="无法执行反思：输入消息为空",
                            name="reflection",
                        )
                    ],
                    "reflection_count": state.get("reflection_count", 0) + 1,
                    "reflection_completed": True
                },
                goto="__end__",
            )
            
        # 检查反思次数，防止无限循环
        max_reflection_count = 3
        current_reflection_count = state.get("reflection_count", 0)
        
        if current_reflection_count >= max_reflection_count:
            logger.warning(f"Reached maximum reflection count ({max_reflection_count}), proceeding to finish")
            return Command(
                update={
                    "messages": [
                        HumanMessage(
                            content=f"已达到最大反思次数({max_reflection_count})，继续执行任务。",
                            name="reflection",
                        )
                    ],
                    "reflection_completed": True
                },
                goto="reporter",
            )
        
        messages = apply_prompt_template("reflection", state)
        
        # Get reflection using reasoning LLM (requires deeper thinking)
        llm = get_llm_by_type("reasoning")
        
        response = llm.invoke(messages)
        
        # 验证响应内容
        if not hasattr(response, "content") or response.content is None:
            logger.error("Reflection returned null content")
            return Command(
                update={
                    "messages": [
                        HumanMessage(
                            content="反思任务失败：返回了空内容",
                            name="reflection",
                        )
                    ],
                    "reflection_count": state.get("reflection_count", 0) + 1,
                    "reflection_completed": True
                },
                goto="__end__",
            )
            
        response_content = response.content
        
        # Check if there are issues/improvements to be made
        needs_revision = "NEEDS_REVISION" in response_content
        
        logger.debug(f"Reflection analysis complete. Needs revision: {needs_revision}")
        logger.debug(f"Reflection content: {response_content}")
        
        # 更新反思计数
        updated_reflection_count = state.get("reflection_count", 0) + 1
        
        if needs_revision:
            # 如果需要修改，添加reflection消息并将其作为下一步的输入
            # 返回给supervisor进行处理
            return Command(
                update={
                    "messages": [
                        HumanMessage(
                            content=response_content,
                            name="reflection",
                        )
                    ],
                    "reflection_count": updated_reflection_count,
                    "needs_revision": True
                },
                goto="reporter",  # 将反思结果交给reporter处理
            )
        else:
            # 如果不需要修改，标记反思已完成，结束流程
            return Command(
                update={
                    "messages": [
                        HumanMessage(
                            content=response_content,
                            name="reflection",
                        )
                    ],
                    "reflection_count": updated_reflection_count,
                    "reflection_completed": True,
                    "needs_revision": False
                },
                goto="__end__",
            )
    except Exception as e:
        logger.error(f"Error in reflection node: {str(e)}")
        # 出现错误时，优雅地失败并进入终止状态
        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content=f"反思过程中出现技术错误：{str(e)}",
                        name="reflection",
                    )
                ],
                "reflection_count": state.get("reflection_count", 0) + 1,
                "reflection_completed": True
            },
            goto="__end__",
        )


def supervisor_node(state: State) -> Command[Literal[*TEAM_MEMBERS, "reflection", "__end__"]]:
    """Supervisor node that decides which agent should act next."""
    logger.info("Supervisor evaluating next action")
    
    # 确保状态中有必要的字段
    if not state.get("messages"):
        logger.warning("No messages in state, ending workflow")
        return Command(goto="__end__")
    
    try:
        # 检查是否已处理过reflection结果
        if state.get("needs_revision") and not state.get("revision_processed"):
            logger.info("Reflection suggested revisions, sending to appropriate agent for corrections")
            
            # 根据需要修改的内容，决定将任务分配给哪个代理
            # 这里可以分析反思消息内容，决定哪个代理最适合处理修改
            # 简单示例：查找关键词决定
            reflection_msg = None
            for msg in reversed(state.get("messages", [])):
                if hasattr(msg, "name") and msg.name == "reflection" and hasattr(msg, "content"):
                    reflection_msg = msg.content.lower()
                    break
            
            # 根据反思内容选择合适的代理来处理
            goto = "reporter"  # 默认发送给reporter进行修改
            
            if reflection_msg:
                if "research" in reflection_msg or "information" in reflection_msg or "search" in reflection_msg:
                    goto = "researcher"
                elif "code" in reflection_msg or "calculation" in reflection_msg or "compute" in reflection_msg:
                    goto = "coder"
                elif "web" in reflection_msg or "browse" in reflection_msg or "website" in reflection_msg:
                    goto = "browser"
            
            # 标记已处理revision建议
            return Command(
                update={"revision_processed": True, "next": goto},
                goto=goto
            )
            
        # 正常的supervisor处理逻辑
        messages = apply_prompt_template("supervisor", state)
        # preprocess messages to make supervisor execute better.
        messages = deepcopy(messages)
        for message in messages:
            if isinstance(message, BaseMessage) and message.name in TEAM_MEMBERS:
                message.content = RESPONSE_FORMAT.format(message.name, message.content)
        
        response = (
            get_llm_by_type(AGENT_LLM_MAP["supervisor"])
            .with_structured_output(schema=Router, method="json_mode")
            .invoke(messages)
        )
        goto = response["next"]
        logger.debug(f"Current state messages: {state['messages']}")
        logger.debug(f"Supervisor response: {response}")

        # Check if we should trigger reflection
        reflection_triggered = False
        max_reflection_count = 3  # 最大反思次数限制
        current_reflection_count = state.get("reflection_count", 0)
        
        if goto == "FINISH":
            # Before finishing, consider if we should reflect on the solution
            if current_reflection_count < max_reflection_count and not state.get("reflection_completed"):
                # 从最近的消息中提取内容进行分析
                last_messages = []
                for msg in state.get("messages", [])[-3:]:
                    if hasattr(msg, "content") and msg.content:
                        last_messages.append(msg.content.lower())
                
                # 仅当有足够的内容可以分析且关键词指示任务即将完成时才触发反思
                if last_messages and any(["final" in msg or "complete" in msg or "conclusion" in msg or "finished" in msg for msg in last_messages]):
                    logger.info("Triggering reflection before completion")
                    reflection_triggered = True
                    goto = "reflection"
            
            if not reflection_triggered:
                goto = "__end__"
                logger.info("Workflow completed")
        else:
            logger.info(f"Supervisor delegating to: {goto}")

        return Command(goto=goto, update={"next": goto})
    
    except Exception as e:
        logger.error(f"Error in supervisor node: {str(e)}")
        # 出现错误时，优雅地失败并进入终止状态
        return Command(goto="__end__")


def planner_node(state: State) -> Command[Literal["supervisor", "__end__"]]:
    """Planner node that generate the full plan."""
    logger.info("Planner generating full plan")
    
    try:
        # 验证输入状态
        if not state.get("messages"):
            logger.warning("No messages in state for planner, ending early")
            return Command(
                update={
                    "messages": [
                        HumanMessage(
                            content="无法执行计划任务：输入消息为空",
                            name="planner",
                        )
                    ],
                    "full_plan": "{}"
                },
                goto="__end__",
            )
            
        messages = apply_prompt_template("planner", state)
        # whether to enable deep thinking mode
        llm = get_llm_by_type("basic")
        if state.get("deep_thinking_mode"):
            llm = get_llm_by_type("reasoning")
            
        # 执行搜索（如果启用）
        if state.get("search_before_planning"):
            try:
                searched_content = tavily_tool.invoke({"query": state["messages"][-1].content})
                if isinstance(searched_content, list):
                    messages = deepcopy(messages)
                    messages[
                        -1
                    ].content += f"\n\n# Relative Search Results\n\n{json.dumps([{'title': elem['title'], 'content': elem['content']} for elem in searched_content], ensure_ascii=False)}"
                else:
                    logger.error(
                        f"Tavily search returned malformed response: {searched_content}"
                    )
            except Exception as e:
                logger.error(f"Error during search before planning: {str(e)}")
                
        # 流式获取响应
        stream = llm.stream(messages)
        full_response = ""
        for chunk in stream:
            if not hasattr(chunk, "content") or chunk.content is None:
                continue
            full_response += chunk.content
            
        logger.debug(f"Current state messages: {state['messages']}")
        logger.debug(f"Planner response: {full_response}")

        # 处理响应格式
        if full_response.startswith("```json"):
            full_response = full_response.removeprefix("```json")

        if full_response.endswith("```"):
            full_response = full_response.removesuffix("```")

        # 验证和修复JSON
        goto = "supervisor"
        try:
            repaired_response = json_repair.loads(full_response)
            full_response = json.dumps(repaired_response)
        except json.JSONDecodeError:
            logger.warning("Planner response is not a valid JSON")
            goto = "__end__"
            full_response = "{\"error\": \"无法解析计划\"}"

        return Command(
            update={
                "messages": [HumanMessage(content=full_response, name="planner")],
                "full_plan": full_response,
            },
            goto=goto,
        )
    except Exception as e:
        logger.error(f"Error in planner node: {str(e)}")
        # 发生错误时提供友好的错误消息
        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content=f"执行计划任务时发生错误: {str(e)}",
                        name="planner",
                    )
                ],
                "full_plan": "{\"error\": \"计划生成失败\"}"
            },
            goto="__end__",
        )


def coordinator_node(state: State) -> Command[Literal["planner", "__end__"]]:
    """Coordinator node that communicate with customers."""
    logger.info("Coordinator talking.")
    
    try:
        # 验证输入状态
        if not state.get("messages"):
            logger.warning("No messages in state for coordinator, ending early")
            return Command(
                update={
                    "messages": [
                        HumanMessage(
                            content="无法执行协调任务：输入消息为空",
                            name="coordinator",
                        )
                    ]
                },
                goto="__end__",
            )
            
        messages = apply_prompt_template("coordinator", state)
        response = get_llm_by_type(AGENT_LLM_MAP["coordinator"]).invoke(messages)
        logger.debug(f"Current state messages: {state['messages']}")
        
        # 验证响应内容
        if not hasattr(response, "content") or response.content is None:
            logger.error("Coordinator returned null content")
            return Command(
                update={
                    "messages": [
                        HumanMessage(
                            content="协调任务失败：返回了空内容",
                            name="coordinator",
                        )
                    ]
                },
                goto="__end__",
            )
            
        response_content = response.content
        # 尝试修复可能的JSON输出
        response_content = repair_json_output(response_content)
        logger.debug(f"Coordinator response: {response_content}")

        goto = "__end__"
        if "handoff_to_planner" in response_content:
            goto = "planner"

        # 更新response.content为修复后的内容
        response.content = response_content

        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content=response_content,
                        name="coordinator",
                    )
                ]
            },
            goto=goto,
        )
    except Exception as e:
        logger.error(f"Error in coordinator node: {str(e)}")
        # 发生错误时提供友好的错误消息
        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content=f"执行协调任务时发生错误: {str(e)}",
                        name="coordinator",
                    )
                ]
            },
            goto="__end__",
        )


def reporter_node(state: State) -> Command[Literal["supervisor"]]:
    """Reporter node that write a final report."""
    logger.info("Reporter write final report")
    
    try:
        # 验证输入状态
        if not state.get("messages"):
            logger.warning("No messages in state for reporter, ending early")
            return Command(
                update={
                    "messages": [
                        HumanMessage(
                            content="无法生成报告：输入消息为空",
                            name="reporter",
                        )
                    ]
                },
                goto="supervisor",
            )
            
        # 检查是否需要处理反思建议
        is_revision = state.get("needs_revision") and state.get("revision_processed")
        
        if is_revision:
            logger.info("Reporter processing reflection feedback")
            # 为reporter准备特殊模板，包括reflection的反馈
            # 获取最近的反思消息
            reflection_msg = None
            for msg in reversed(state.get("messages", [])):
                if hasattr(msg, "name") and msg.name == "reflection" and hasattr(msg, "content"):
                    reflection_msg = msg.content
                    break
                    
            if reflection_msg:
                # 创建一个临时状态，包含反思信息
                temp_state = deepcopy(state)
                temp_state["reflection_feedback"] = reflection_msg
                messages = apply_prompt_template("reporter_revision", temp_state)
                
                # 如果reporter_revision模板不存在，使用普通模板
                if not messages or (isinstance(messages, list) and len(messages) == 1 and "Error in template processing" in messages[0].get("content", "")):
                    logger.warning("reporter_revision template not found, using standard reporter template")
                    messages = apply_prompt_template("reporter", state)
            else:
                messages = apply_prompt_template("reporter", state)
        else:
            # 正常报告生成
            messages = apply_prompt_template("reporter", state)
            
        response = get_llm_by_type(AGENT_LLM_MAP["reporter"]).invoke(messages)
        logger.debug(f"Current state messages: {state['messages']}")
        
        # 验证响应内容
        if not hasattr(response, "content") or response.content is None:
            logger.error("Reporter returned null content")
            return Command(
                update={
                    "messages": [
                        HumanMessage(
                            content="报告生成失败：返回了空内容",
                            name="reporter",
                        )
                    ]
                },
                goto="supervisor",
            )
        
        response_content = response.content
        # 尝试修复可能的JSON输出
        response_content = repair_json_output(response_content)
        logger.debug(f"reporter response: {response_content}")

        # 添加状态更新，清除反思相关标记
        updates = {
            "messages": [
                HumanMessage(
                    content=response_content,
                    name="reporter",
                )
            ]
        }
        
        # 如果这是修订版本，标记修订已完成
        if is_revision:
            updates["revision_processed"] = True
            updates["needs_revision"] = False
            updates["reflection_completed"] = True
            logger.info("Report revision completed")

        return Command(
            update=updates,
            goto="supervisor",
        )
    except Exception as e:
        logger.error(f"Error in reporter node: {str(e)}")
        # 发生错误时提供友好的错误消息
        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content=f"生成报告时发生错误: {str(e)}",
                        name="reporter",
                    )
                ]
            },
            goto="supervisor",
        )
