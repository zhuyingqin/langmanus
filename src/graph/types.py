from typing import Literal, Optional
from typing_extensions import TypedDict
from langgraph.graph import MessagesState

from src.config import TEAM_MEMBERS

# Define routing options
OPTIONS = TEAM_MEMBERS + ["FINISH"]


class Router(TypedDict, total=False):
    """Worker to route to next. If no workers needed, route to FINISH."""

    next: Literal[*OPTIONS]
    browser_instruction: Optional[str]  # 可选的浏览器指令字段


class State(MessagesState):
    """State for the agent system, extends MessagesState with next field."""

    # Constants
    TEAM_MEMBERS: list[str]

    # Runtime Variables
    next: Optional[str] = None
    full_plan: Optional[str] = None
    deep_thinking_mode: bool = False
    search_before_planning: bool = False
    reflection_count: int = 0
    browser_instruction: Optional[str] = None  # 浏览器指令
    current_instruction: Optional[str] = None  # 当前正在执行的指令
    needs_revision: bool = False  # 是否需要修订
    revision_processed: bool = False  # 修订是否已处理
    reflection_completed: bool = False  # 反思是否已完成
