import logging
from src.config import TEAM_MEMBERS
from src.graph import build_graph
from src.utils.log_handler import setup_logging, enable_debug_logging

# 创建全局日志记录器
logger = logging.getLogger(__name__)

# 创建工作流图
graph = build_graph()


def run_agent_workflow(user_input: str, debug: bool = False, save_logs: bool = True):
    """Run the agent workflow with the given user input.

    Args:
        user_input: The user's query or request
        debug: If True, enables debug level logging
        save_logs: If True, saves logs to file

    Returns:
        The final state after the workflow completes
    """
    if not user_input:
        raise ValueError("Input could not be empty")

    # 设置日志记录
    setup_logging(debug=debug, save_to_file=save_logs)
    
    if debug:
        enable_debug_logging()

    logger.info(f"Starting workflow with user input: {user_input}")
    result = graph.invoke(
        {
            # Constants
            "TEAM_MEMBERS": TEAM_MEMBERS,
            # Runtime Variables
            "messages": [{"role": "user", "content": user_input}],
            "deep_thinking_mode": True,
            "search_before_planning": True,
        }
    )
    logger.debug(f"Final workflow state: {result}")
    logger.info("Workflow completed successfully")
    return result


if __name__ == "__main__":
    print(graph.get_graph().draw_mermaid())
