import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

from src.schemas import AgentState
from src.tools import search_web
from src.nodes import (
    agent_node,
    should_continue,
    human_approval_node,
    check_approval,
    handle_rejection
)

load_dotenv()


def create_demo_agent():
    """Creates a LangGraph agent with Gemini and web search capability.
    
    All tools require human approval before execution. The agent uses an
    interrupt-based approval flow that allows for repetitive editing.
    
    Returns:
        CompiledGraph: Compiled agent graph with memory
    """
    model = init_chat_model(
        model="gemini-2.5-flash-lite",
        model_provider="google_genai",
        api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0
    )
    
    tools = [search_web]
    model_with_tools = model.bind_tools(tools)
    
    workflow = StateGraph(AgentState)
    
    workflow.add_node("agent", lambda state: agent_node(state, model_with_tools))
    workflow.add_node("approval", human_approval_node)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("rejected", handle_rejection)
    
    workflow.set_entry_point("agent")
    
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "approval": "approval",
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "approval",
        check_approval,
        {
            "tools": "tools",
            "rejected": "rejected"
        }
    )
    
    workflow.add_edge("tools", "agent")
    workflow.add_edge("rejected", "agent")
    
    memory = MemorySaver()
    compiled_graph = workflow.compile(checkpointer=memory)
    
    return compiled_graph

