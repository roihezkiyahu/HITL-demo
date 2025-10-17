import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command

from src.agent import create_demo_agent

load_dotenv()


def is_tool_approval_request(interrupt_data):
    """Checks if the interrupt is a tool approval request.
    
    Args:
        interrupt_data: Interrupt data from the graph
        
    Returns:
        bool: True if this is a tool approval request
    """
    for item in interrupt_data:
        if isinstance(item.value, dict) and "tool_calls" in item.value:
            return True
    return False


def extract_tool_calls(interrupt_data):
    """Extracts tool calls from interrupt data.
    
    Args:
        interrupt_data: Interrupt data from the graph
        
    Returns:
        list: List of tool calls
    """
    for item in interrupt_data:
        if isinstance(item.value, dict) and "tool_calls" in item.value:
            return item.value["tool_calls"]
    return []


def handle_tool_approval(interrupt_data):
    """Handles the tool approval process by prompting the user.
    
    Args:
        interrupt_data: Interrupt data containing tool call information
        
    Returns:
        dict: Approval response with approved status and optional feedback
    """
    tool_calls = extract_tool_calls(interrupt_data)
    
    print("\n" + "="*60)
    print("TOOL CALL APPROVAL REQUIRED")
    print("="*60)
    
    for i, tool_call in enumerate(tool_calls, 1):
        print(f"\nTool Call #{i}:")
        print(f"  Tool: {tool_call.get('name')}")
        print(f"  Arguments: {tool_call.get('args')}")
    
    print("\n" + "-"*60)
    approval = input("Approve these tool calls? (yes/no): ").strip().lower()
    
    if approval in ["yes", "y"]:
        print("✓ Approved")
        return {"approved": True}
    else:
        feedback = input("Provide feedback for the agent: ").strip()
        if not feedback:
            feedback = "The user rejected the tool call. Please revise your approach."
        
        print(f"✗ Rejected with feedback: {feedback}")
        return {
            "approved": False,
            "feedback": feedback
        }


def process_interrupts(agent, result, config):
    """Processes workflow interrupts until completion.
    
    Args:
        agent: Compiled agent graph
        result: Current agent result
        config: Agent configuration dictionary
        
    Returns:
        dict: Final result after processing all interrupts
    """
    while "__interrupt__" in result:
        interrupt_data = result["__interrupt__"]
        
        if is_tool_approval_request(interrupt_data):
            approval_response = handle_tool_approval(interrupt_data)
            result.pop("__interrupt__")
            result = agent.invoke(
                Command(update=approval_response, resume=approval_response),
                config=config
            )
        else:
            print(f"Unknown interrupt type: {interrupt_data}")
            break
    
    return result


def extract_final_message(result):
    """Extracts the final message content from agent result.
    
    Args:
        result: Agent result dictionary
        
    Returns:
        str: Final message content
    """
    final_message = result["messages"][-1]
    
    if isinstance(final_message, AIMessage):
        return final_message.content
    
    return str(final_message)


def main():
    """Main CLI loop for the HITL demo."""
    print("\n" + "="*60)
    print("Human-in-the-Loop Agent Demo (CLI)")
    print("="*60)
    print("\nThis agent uses Google Gemini with web search capability.")
    print("All tool calls require your approval before execution.")
    print("You can reject and provide feedback for repetitive editing.")
    print("\nType 'quit' or 'exit' to stop.\n")
    
    if not os.getenv("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY not found in environment variables.")
        print("Please set your Google API key in a .env file or environment.")
        return
    
    print("Initializing agent...")
    agent = create_demo_agent()
    print("✓ Agent initialized\n")
    
    thread_id = "cli_session"
    config = {"configurable": {"thread_id": thread_id}}
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "approved": False,
            "feedback": ""
        }
        
        try:
            result = agent.invoke(initial_state, config=config)
            result = process_interrupts(agent, result, config)
            
            response = extract_final_message(result)
            print(f"\nAgent: {response}\n")
            
        except Exception as e:
            print(f"\nError: {str(e)}\n")


if __name__ == "__main__":
    main()

