import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command

from src.agent import create_demo_agent

load_dotenv()


def initialize_session_state():
    """Initializes Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "agent" not in st.session_state:
        st.session_state.agent = None
    
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = "streamlit_session"
    
    if "pending_approval" not in st.session_state:
        st.session_state.pending_approval = None
    
    if "agent_config" not in st.session_state:
        st.session_state.agent_config = None
    
    if "agent_result" not in st.session_state:
        st.session_state.agent_result = None
    
    if "show_feedback" not in st.session_state:
        st.session_state.show_feedback = False


def initialize_agent():
    """Initializes the agent if not already initialized."""
    try:
        if not os.getenv("GOOGLE_API_KEY"):
            st.error("GOOGLE_API_KEY not found. Please set it in your .env file.")
            return False
        
        st.session_state.agent = create_demo_agent()
        return True
        
    except Exception as e:
        st.error(f"Failed to initialize agent: {str(e)}")
        return False


def is_tool_approval_request(interrupt_data):
    """Checks if interrupt is a tool approval request."""
    for item in interrupt_data:
        if isinstance(item.value, dict) and "tool_calls" in item.value:
            return True
    return False


def extract_tool_calls(interrupt_data):
    """Extracts tool calls from interrupt data."""
    for item in interrupt_data:
        if isinstance(item.value, dict) and "tool_calls" in item.value:
            return item.value["tool_calls"]
    return []


def process_agent_response(result):
    """Processes agent result and extracts the final message.
    
    Args:
        result: Agent result dictionary
        
    Returns:
        str or None: Final message content, or None if interrupted
    """
    if "__interrupt__" in result:
        interrupt_data = result["__interrupt__"]
        
        if is_tool_approval_request(interrupt_data):
            tool_calls = extract_tool_calls(interrupt_data)
            st.session_state.pending_approval = tool_calls
            st.session_state.agent_result = result
            return None
    
    st.session_state.pending_approval = None
    st.session_state.agent_result = None
    
    final_message = result["messages"][-1]
    
    if isinstance(final_message, AIMessage):
        return final_message.content
    
    return str(final_message)


def send_message(user_input: str):
    """Sends a message to the agent and processes the response."""
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    st.session_state.agent_config = config
    
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "approved": False,
        "feedback": ""
    }
    
    try:
        result = st.session_state.agent.invoke(initial_state, config=config)
        response = process_agent_response(result)
        
        if response:
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    except Exception as e:
        st.error(f"Error: {str(e)}")


def handle_approval(approved: bool, feedback: str = ""):
    """Handles approval or rejection of tool calls."""
    if not st.session_state.agent_result or not st.session_state.agent_config:
        return
    
    if approved:
        approval_response = {"approved": True}
    else:
        if not feedback:
            feedback = "The user rejected the tool call. Please revise your approach."
        approval_response = {"approved": False, "feedback": feedback}
    
    result = st.session_state.agent_result
    result.pop("__interrupt__", None)
    
    try:
        result = st.session_state.agent.invoke(
            Command(update=approval_response, resume=approval_response),
            config=st.session_state.agent_config
        )
        
        response = process_agent_response(result)
        
        if response:
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    except Exception as e:
        st.error(f"Error: {str(e)}")


def clear_conversation():
    """Clears the conversation history and resets state."""
    import uuid
    
    st.session_state.messages = []
    st.session_state.pending_approval = None
    st.session_state.agent_result = None
    st.session_state.agent_config = None
    st.session_state.show_feedback = False
    st.session_state.thread_id = f"streamlit_session_{uuid.uuid4().hex[:8]}"


def display_approval_ui():
    """Displays the approval UI for pending tool calls."""
    st.warning("⚠️ Tool Call Approval Required")
    
    tool_calls = st.session_state.pending_approval
    
    for i, tool_call in enumerate(tool_calls, 1):
        with st.expander(f"Tool Call #{i}: {tool_call.get('name')}", expanded=True):
            st.json(tool_call.get('args', {}))
    
    if not st.session_state.show_feedback:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✅ Approve", use_container_width=True, type="primary"):
                handle_approval(True)
                st.rerun()
        
        with col2:
            if st.button("❌ Reject", use_container_width=True):
                st.session_state.show_feedback = True
                st.rerun()
    else:
        feedback = st.text_area(
            "Provide feedback for the agent:",
            placeholder="Explain why you rejected and what you want instead...",
            height=100
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Submit Rejection", use_container_width=True, type="primary"):
                handle_approval(False, feedback)
                st.session_state.show_feedback = False
                st.rerun()
        
        with col2:
            if st.button("Cancel", use_container_width=True):
                st.session_state.show_feedback = False
                st.rerun()


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="HITL Agent Demo",
        page_icon="🤖",
        layout="centered"
    )
    
    initialize_session_state()
    
    st.title("🤖 Human-in-the-Loop Agent Demo")
    st.caption("Powered by Google Gemini with web search capability")
    
    with st.sidebar:
        st.header("About")
        st.write("""
        This demo showcases a Human-in-the-Loop agent that:
        - Requires approval before executing tools
        - Allows repetitive editing via rejection feedback
        - Maintains conversation context
        """)
        
        st.divider()
        
        if st.session_state.agent:
            st.success("✓ Agent Ready")
        else:
            if st.button("Initialize Agent", type="primary"):
                if initialize_agent():
                    st.success("Agent initialized!")
                    st.rerun()
        
        if st.button("Clear Chat History"):
            clear_conversation()
            st.rerun()
    
    if not st.session_state.agent:
        st.info("👈 Please initialize the agent from the sidebar to start.")
        return
    
    if st.session_state.pending_approval:
        display_approval_ui()
        st.divider()
    
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
    
    if not st.session_state.pending_approval:
        user_input = st.chat_input("Type your message here...")
        
        if user_input:
            with st.spinner("Processing..."):
                send_message(user_input)
            st.rerun()


if __name__ == "__main__":
    main()

