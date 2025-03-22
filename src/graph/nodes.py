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
    result = research_agent.invoke(state)
    logger.info("Research agent completed task")
    response_content = result["messages"][-1].content
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


def code_node(state: State) -> Command[Literal["supervisor"]]:
    """Node for the coder agent that executes Python code."""
    logger.info("Code agent starting task")
    result = coder_agent.invoke(state)
    logger.info("Code agent completed task")
    response_content = result["messages"][-1].content
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


def browser_node(state: State) -> Command[Literal["supervisor"]]:
    """Node for the browser agent that performs web browsing tasks."""
    logger.info("Browser agent starting task")
    result = browser_agent.invoke(state)
    logger.info("Browser agent completed task")
    response_content = result["messages"][-1].content
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


def reflection_node(state: State) -> Command[Literal["supervisor", "__end__"]]:
    """反思节点，批评和改进当前解决方案。
    
    该节点分析当前状态，包括所有消息和迄今为止的决策，
    以识别当前方法的潜在改进或问题。
    """
    logger.info("Reflection agent starting analysis")
    
    # 确保state中有消息
    if not state.get("messages"):
        logger.warning("No messages found in state, skipping reflection")
        return Command(
            update={"reflection_count": 1},
            goto="__end__",
        )
    
    messages = apply_prompt_template("reflection", state)
    
    # Get reflection using reasoning LLM (requires deeper thinking)
    llm = get_llm_by_type("reasoning")
    
    try:
        response = llm.invoke(messages)
        response_content = response.content
        
        # Check if there are issues/improvements to be made
        needs_revision = "NEEDS_REVISION" in response_content
        
        logger.debug(f"Reflection analysis complete. Needs revision: {needs_revision}")
        logger.debug(f"Reflection content: {response_content}")
        
        if needs_revision:
            # If improvements needed, add reflection message and return to supervisor
            return Command(
                update={
                    "messages": [
                        HumanMessage(
                            content=response_content,
                            name="reflection",
                        )
                    ],
                    "reflection_count": state.get("reflection_count", 0) + 1,
                },
                goto="supervisor",
            )
        else:
            # If no improvements needed, proceed to end
            return Command(
                update={
                    "messages": [
                        HumanMessage(
                            content=response_content,
                            name="reflection",
                        )
                    ],
                    "reflection_count": state.get("reflection_count", 0) + 1,
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
                        content="反思过程中出现技术错误，跳过反思阶段。",
                        name="reflection",
                    )
                ],
                "reflection_count": state.get("reflection_count", 0) + 1,
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
            if current_reflection_count < max_reflection_count:
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
    messages = apply_prompt_template("planner", state)
    # whether to enable deep thinking mode
    llm = get_llm_by_type("basic")
    if state.get("deep_thinking_mode"):
        llm = get_llm_by_type("reasoning")
    if state.get("search_before_planning"):
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
    stream = llm.stream(messages)
    full_response = ""
    for chunk in stream:
        full_response += chunk.content
    logger.debug(f"Current state messages: {state['messages']}")
    logger.debug(f"Planner response: {full_response}")

    if full_response.startswith("```json"):
        full_response = full_response.removeprefix("```json")

    if full_response.endswith("```"):
        full_response = full_response.removesuffix("```")

    goto = "supervisor"
    try:
        repaired_response = json_repair.loads(full_response)
        full_response = json.dumps(repaired_response)
    except json.JSONDecodeError:
        logger.warning("Planner response is not a valid JSON")
        goto = "__end__"

    return Command(
        update={
            "messages": [HumanMessage(content=full_response, name="planner")],
            "full_plan": full_response,
        },
        goto=goto,
    )


def coordinator_node(state: State) -> Command[Literal["planner", "__end__"]]:
    """Coordinator node that communicate with customers."""
    logger.info("Coordinator talking.")
    messages = apply_prompt_template("coordinator", state)
    response = get_llm_by_type(AGENT_LLM_MAP["coordinator"]).invoke(messages)
    logger.debug(f"Current state messages: {state['messages']}")
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


def reporter_node(state: State) -> Command[Literal["supervisor"]]:
    """Reporter node that write a final report."""
    logger.info("Reporter write final report")
    messages = apply_prompt_template("reporter", state)
    response = get_llm_by_type(AGENT_LLM_MAP["reporter"]).invoke(messages)
    logger.debug(f"Current state messages: {state['messages']}")
    response_content = response.content
    # 尝试修复可能的JSON输出
    response_content = repair_json_output(response_content)
    logger.debug(f"reporter response: {response_content}")

    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=response_content,
                    name="reporter",
                )
            ]
        },
        goto="supervisor",
    )
