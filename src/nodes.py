from typing import Literal
from langgraph.types import interrupt
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import END
from src.schemas import AgentState
from src.prompts import get_system_prompt

def agent_node(state: AgentState, model):
    """Calls the language model with current messages.
    
    Args:
        state: Current agent state
        model: Language model with tools bound
        
    Returns:
        dict: Updated state with model response
    """
    messages = state["messages"]
    
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=get_system_prompt())] + messages
    
    response = model.invoke(messages)
    
    if not response.content and not getattr(response, 'tool_calls', None):
        from langchain_core.messages import ToolMessage
        has_tool_message = any(isinstance(msg, ToolMessage) for msg in messages)
        
        if has_tool_message:
            prompt_message = HumanMessage(content="Please provide a summary of the search results above.")
            retry_response = model.invoke(messages + [prompt_message])
            return {"messages": [response, prompt_message, retry_response], "approved": False}
    
    return {"messages": [response], "approved": False}


def should_continue(state: AgentState) -> str:
    """Determines if agent should continue to approval or end.
    
    Args:
        state: Current agent state
        
    Returns:
        str: Next node ("approval" or END)
    """
    
    messages = state["messages"]
    last_message = messages[-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "approval"
    
    return END


def human_approval_node(state: AgentState):
    """Interrupts execution to request human approval for tool calls.
    
    Args:
        state: Current agent state
        
    Returns:
        dict: State with approval status and feedback
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        tool_calls_info = []
        for tool_call in last_message.tool_calls:
            tool_calls_info.append({
                "name": tool_call.get("name"),
                "args": tool_call.get("args", {}),
                "id": tool_call.get("id")
            })
        
        approval_data = interrupt({
            "tool_calls": tool_calls_info,
            "message": "Tool call(s) require approval"
        })
        
        approved = approval_data.get("approved", False)
        feedback = approval_data.get("feedback", "The user rejected the tool call. Please revise your approach.")
        
        return {"approved": approved, "feedback": feedback}
    
    return {"approved": False, "feedback": ""}


def check_approval(state: AgentState) -> Literal["tools", "rejected"]:
    """Checks if tool calls were approved or rejected.
    
    Args:
        state: Current agent state
        
    Returns:
        str: Next node ("tools" if approved, "rejected" if not)
    """
    if state.get("approved", False):
        return "tools"
    else:
        return "rejected"


def handle_rejection(state: AgentState):
    """Handles tool call rejection by injecting feedback.
    
    Args:
        state: Current agent state
        
    Returns:
        dict: Updated state with feedback message
    """
    feedback = state.get("feedback", "The user rejected the tool call. Please revise your approach.")
    feedback_message = HumanMessage(content=feedback)
    
    return {"messages": [feedback_message], "approved": False}

