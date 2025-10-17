# Building Multi-Turn Edit Loops in Tool-Based LangGraph Agents

## Introduction: Beyond Single Interrupts

Most Human-in-the-Loop (HITL) tutorials stop at the same place: an agent pauses, asks for human approval, and then either proceeds or fails. Single interrupt patterns work fine for one-shot decisions, but they break down in real workflows where users need to **refine, iterate, and correct** tool inputs across multiple turns.

Consider a common scenario: you're building an AI agent that searches the web for information. The agent decides to call a search tool with `backend="serp"`, but you don't have a SERP API key configured. What happens next? Most single-interrupt HITL systems either fail or let the tool error out. The conversation ends.

With a **multi-turn edit loop**, you can reject that tool call, provide feedback like *"Use DuckDuckGo instead"*, and the agent regenerates its decision with that constraint in mind. If there's still an issue, you reject again. And again. The loop persists until you get it right—or you decide to stop.

This article explains how to build that pattern in LangGraph. We'll ground everything in a real, working implementation: a web search agent that requires human approval before any tool execution and allows unlimited rejection-and-retry cycles.

---

## Background: Understanding Interrupt Patterns

### The Single-Interrupt Pattern

Most HITL tutorials follow this flow:

```
[Agent Generates Tool Call]
           ↓
    [INTERRUPT: Ask Human]
           ↓
    [Approve or Reject?]
           ↓
   [Execute Tool or Exit]
           ↓
      [End]
```

The pattern is simple: pause the graph at a decision point, collect human input, then proceed or terminate. This works for scenarios like content moderation ("Should this post be deleted?") or cost estimation ("Do you authorize this $50 API call?").

But notice what happens if the human rejects: the conversation typically ends. If they want to retry with different parameters, they start over. There's no mechanism for the agent to learn from the rejection and self-correct.

### The Multi-Turn Edit Loop Pattern

Our implementation introduces a crucial change:

```
[Agent Generates Tool Call]
           ↓
    [INTERRUPT: Ask Human]
           ↓
    [Approve or Reject?]
           ↓
      [APPROVED?]
       /        \
      ✓          ✗
      |          |
   [Execute   [Inject Feedback
    Tool]      as Message]
      |          |
      |    [Loop Back to Agent]
      |          |
      └────→ [Agent Regenerates]
             [with Feedback]
```

The key difference: **rejection loops back to the agent**, not to the user. The agent sees the rejection as a new human message in the conversation context and generates a revised tool call. The human approves or rejects again—creating an iterative refinement loop.

This is elegant because it preserves conversation context, allows the LLM to self-correct, and creates a natural UX where users guide the agent through multiple attempts.

---

## Architecture Overview

Let's examine the actual implementation. The graph structure in `src/agent.py` defines four nodes:

1. **agent** — Calls the LLM with tools
2. **approval** — Interrupts to request human decision
3. **tools** — Executes approved tool calls
4. **rejected** — Handles rejections by injecting feedback

The critical edges for multi-turn looping are:

- `agent` → `approval` (conditional: if tool calls exist)
- `approval` → `tools` (if approved)
- `approval` → `rejected` (if rejected)
- `tools` → `agent` (loop back)
- `rejected` → `agent` (loop back with feedback)

Notice the two edges feeding back into `agent`: after tool execution OR after rejection. This creates the multi-turn loop.

The state schema (from `src/schemas.py`) is minimal but sufficient:

```python
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    approved: bool
    feedback: str
```

Three fields:
- **messages**: The conversation history (system, user, assistant, tool messages)
- **approved**: Boolean flag for approval status
- **feedback**: Optional user feedback for rejections

This simplicity is key. We don't need complex iteration counters or history tracking—the graph structure and message accumulation handle everything.

---

## Implementation Deep-Dive

### The Agent Node

The agent node calls the language model with bound tools:

```python
def agent_node(state: AgentState, model):
    messages = state["messages"]
    
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=get_system_prompt())] + messages
    
    response = model.invoke(messages)
    
    if not response.content and not getattr(response, 'tool_calls', None):
        has_tool_message = any(isinstance(msg, ToolMessage) for msg in messages)
        
        if has_tool_message:
            prompt_message = HumanMessage(content="Please provide a summary of the search results above.")
            retry_response = model.invoke(messages + [prompt_message])
            return {"messages": [response, prompt_message, retry_response], "approved": False}
    
    return {"messages": [response], "approved": False}
```

Key points:
- Ensures a system message is present (consistent agent behavior)
- Handles edge case where model returns tool results but no content
- Returns a new message appended to the state (thanks to `add_messages`, it accumulates)

### The Approval Node: Interrupting for Human Decision

The approval node is where the magic happens:

```python
def human_approval_node(state: AgentState):
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
```

The `interrupt()` call from LangGraph pauses execution and sends data to the client. The client responds with approval status and optional feedback, which gets returned and stored in state.

### The Rejection Handler: Closing the Loop

This is the secret to multi-turn refinement:

```python
def handle_rejection(state: AgentState):
    feedback = state.get("feedback", "The user rejected the tool call. Please revise your approach.")
    feedback_message = HumanMessage(content=feedback)
    
    return {"messages": [feedback_message], "approved": False}
```

It's simple: take the user's feedback and inject it back into the conversation as a HumanMessage. Now the agent sees:

1. Its own tool call (in the previous AIMessage)
2. The human's feedback (in the new HumanMessage)

On the next invocation, the agent regenerates with this context. If the feedback was "Use DuckDuckGo instead of SERP," the agent will try that. If it still doesn't work, the human rejects again, feedback accumulates, and the loop continues.

---

## A Concrete Multi-Turn Example

Let's trace a realistic scenario:

### Round 1: Initial Request

**User:** "What's the latest news on AI safety?"

**Agent Thinks:** I need to search the web. I'll use the SERP backend for quality results.

**Agent Tool Call:**
```json
{
  "name": "search_web",
  "args": {
    "queries": ["AI safety latest news 2025"],
    "backend": "serp",
    "num_results": 5
  }
}
```

**Human Reviews:** Hmm, SERP requires an API key I don't have configured.

**Human Rejects:** `approved=false, feedback="I don't have a SERP API key. Please use the free DuckDuckGo backend instead."`

### Round 2: Agent Corrects

The rejection feedback is injected as a HumanMessage. The agent sees its previous attempt plus the correction request.

**Agent Thinks:** Oh, I tried SERP but the user doesn't have that API key. They want DuckDuckGo. Let me try again.

**Agent Tool Call:**
```json
{
  "name": "search_web",
  "args": {
    "queries": ["AI safety latest news 2025"],
    "backend": "duckduckgo",
    "num_results": 5
  }
}
```

**Human Reviews:** Good! This looks right.

**Human Approves:** `approved=true`

### Round 3: Execution

The approved tool call executes, results come back as a ToolMessage, and the agent generates a summary from the results.

**Agent Output:** "Based on the latest search results, here are the key AI safety developments..."

**Conversation Ends:** User is satisfied, agent stops.

If the human rejected the DuckDuckGo search too, the loop would continue indefinitely (though in practice, users provide better feedback or manually stop).

---

## State Management and Persistence

A critical aspect many HITL implementations miss: **how does state persist across approvals?**

The answer is LangGraph's checkpoint system. In `src/agent.py`:

```python
memory = MemorySaver()
compiled_graph = workflow.compile(checkpointer=memory)
```

`MemorySaver` stores the graph state by thread ID. When you call:

```python
config = {"configurable": {"thread_id": "session_123"}}
result = agent.invoke(initial_state, config=config)
```

LangGraph saves the state with that thread ID. When you interrupt and then resume (after the human decision), LangGraph retrieves the checkpoint, resumes execution, and continues seamlessly.

This means:
- **Conversation history is preserved** across approval/rejection cycles
- **Tool calls are never executed twice** (because you only resume after human decision)
- **Multiple sessions can run in parallel** (each with its own thread ID)

For the CLI (`cli_demo.py`), a single thread ID is used per session. For Streamlit (`streamlit_demo.py`), each session gets a unique thread ID. This allows the same agent to handle multiple concurrent users.

---

## UI Implementations: How Humans Interact

The multi-turn loop works the same regardless of UI, but the experience differs:

### CLI Implementation

The `process_interrupts()` function in `cli_demo.py` handles multiple approval cycles:

```python
def process_interrupts(agent, result, config):
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
```

The loop is explicit: while there's an interrupt, ask the human, get their response, and invoke the agent again with the approval data. This handles multiple approval cycles in a single user turn.

### Streamlit Implementation

The Streamlit UI (`streamlit_demo.py:120-147`) handles approvals differently because it's stateful:

```python
def handle_approval(approved: bool, feedback: str = ""):
    if not st.session_state.agent_result or not st.session_state.agent_config:
        return
    
    if approved:
        approval_response = {"approved": True}
    else:
        if not feedback:
            feedback = "The user rejected the tool call. Please revise your approach."
        approval_response = {"approved": False, "feedback": feedback}
    
    result = st.session_state.agent_result
    result.pop("__interrupt__")
    
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
```

When the user clicks "Approve" or "Reject" (with feedback), this function resumes the graph with the approval data. If the agent generates another tool call, the UI interrupts again and displays another approval prompt. The loop happens within Streamlit's page renders and reruns.

Both implementations achieve the same multi-turn loop—just adapted to their execution model (CLI blocks synchronously; Streamlit rerenders asynchronously).

---

## Challenges and Design Decisions

Building multi-turn loops introduces real tradeoffs:

### 1. **No Built-In Loop Limit**

The current implementation allows infinite rejection cycles. A user could theoretically keep rejecting forever, creating an infinite loop. Solutions:

- Add an explicit counter: `state["round_count"]` incremented on each rejection
- Add a timeout: use `interrupt()` with a timeout parameter
- Add cost tracking: abort after too many tool invocations

The current implementation trusts users to stop, which is reasonable for low-risk scenarios (web search) but risky for high-cost operations (API calls, model training).

### 2. **Feedback Quality**

The multi-turn loop is only as good as the feedback. If a user rejects with vague feedback ("This is wrong"), the agent might not know how to fix it. Better feedback ("Use backend=duckduckgo instead of serp") yields better corrections.

You could improve this with:
- Structured feedback forms (checkboxes, dropdowns)
- Examples of good feedback
- Automatic suggestion generation (agent proposes corrections)

### 3. **Tool Side Effects**

The current implementation only executes tools after approval, so there's no risk of repeated tool invocations. But if you allowed intermediate tool execution (e.g., for debugging), you'd need idempotency guarantees or rollback mechanisms.

### 4. **Message Accumulation**

Messages accumulate in state with each loop turn. For long approval cycles, this could exceed token limits. Solutions:

- Summarize old messages periodically
- Archive messages to external storage
- Implement a sliding window of recent messages

The current implementation is stateless in this regard—if it becomes an issue, these are easy retrofits.

---

## Practical Takeaways

After building and analyzing this pattern, here are the key lessons:

### 1. **Rejection Requires a Return Edge**

The critical architectural insight: if you want multi-turn loops, you need an explicit edge from rejection back to the agent. Without it, rejection terminates.

```python
workflow.add_edge("rejected", "agent")
```

That single line transforms single-interrupt into multi-turn.

### 2. **Feedback Must Be Injected as Messages**

Don't store feedback in a separate field and hope the LLM sees it. Inject it directly into the conversation:

```python
feedback_message = HumanMessage(content=feedback)
return {"messages": [feedback_message], "approved": False}
```

Now it's part of the conversation context. The LLM naturally understands it's a correction request.

### 3. **State Persistence Is Non-Negotiable**

Without checkpointing, multi-turn loops break. You lose conversation history between approval cycles. Always use a checkpointer:

```python
memory = MemorySaver()  # or PGCheckpointer, SQLCheckpointer, etc.
compiled_graph = workflow.compile(checkpointer=memory)
```

### 4. **Generalize Beyond Tool Approval**

This pattern isn't specific to tool calls. It generalizes to any refinement loop:

- **Code generation**: Generate code → human reviews → provides feedback → agent refines
- **Content creation**: Write article → editor reviews → provides notes → writer revises
- **Data extraction**: Extract fields → human verifies → flags errors → agent re-extracts
- **API orchestration**: Plan API sequence → human questions order → suggests changes → agent replans

The architecture stays the same; only the nodes and feedback types differ.

---

## Conclusion: Iterative Refinement as a First-Class Pattern

Single-interrupt HITL systems treat human involvement as a gating mechanism: pause, ask, proceed or abort. But real workflows are iterative. Users need to steer, adjust, and refine.

The multi-turn edit loop pattern—enabled by a simple architectural change (rejection looping to agent)—transforms HITL systems from binary gates into interactive refinement engines. Users don't just approve or reject; they guide the agent through multiple attempts toward the desired outcome.

Building this required:
- A clear graph structure with explicit feedback edges
- State management to track conversation history
- Feedback injection as conversation messages
- Checkpoint persistence across approval cycles

The implementation in this codebase demonstrates that you don't need elaborate systems. A simple 4-node graph, a basic state schema, and proper message handling unlock powerful multi-turn interaction.

If you're building tool-based agents, consider this pattern early. It's a small architectural change with outsized UX benefits. Whether you're generating code, extracting data, orchestrating APIs, or searching the web—**iteration is essential**. Make it a first-class feature of your graph.

---

## References and Further Exploration

- **LangGraph Documentation**: https://langchain-ai.github.io/langgraph/
- **Interrupt & Command API**: Key to implementing resumable flows
- **Checkpointing**: Essential for multi-turn conversation persistence
- **Tool Binding**: How LangChain agents bind tools to language models

The code for this implementation is available in the repository. Explore `src/agent.py` for the graph definition, `src/nodes.py` for node implementations, and both `cli_demo.py` and `streamlit_demo.py` for UI-specific patterns.

Happy building.
