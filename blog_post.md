# Building Multi-Turn Edit Loops in Tool-Based LangGraph Agents

## Introduction: Beyond Single Interrupts

Most Human-in-the-Loop (HITL) tutorials stop at the same place: an agent pauses, asks for human approval, and then either proceeds or fails. Single interrupt patterns work fine for one-shot decisions, but they break down in real workflows where users need to refine, iterate, and correct tool inputs across multiple turns.

Consider a common scenario: you're building an AI agent that searches the web for information. The agent decides to call a search tool with `backend="serp"`, but you don't have a SERP API key configured. What happens next? Most single-interrupt HITL systems either fail or let the tool error out. The conversation ends.

But the stakes get higher with destructive tools. Imagine an agent that deletes database records, modifies files, or executes API calls with billing implications. A single-interrupt system forces a binary choice: approve and execute, or reject and start over. There's no room for "wait, let me check that parameter first" or "I think you should delete from the staging table, not production." If rejection kills the conversation, users cannot refine their intent. With destructive operations, that's not just inconvenient, it's dangerous.

With a multi-turn edit loop, you can reject that tool call, provide feedback like "Use DuckDuckGo instead" or "Delete only records matching status=inactive", and the agent regenerates its decision with that constraint in mind. If there's still an issue, you reject again. And again. The loop persists until you get it right, or you decide to stop. For dangerous operations, this iterative refinement is essential.

This article explains how to build that pattern in LangGraph. We'll ground everything in a real, working implementation: a web search agent that requires human approval before any tool execution and allows unlimited rejection and retry cycles.

---

## Background: Understanding Interrupt Patterns

### The Single-Interrupt Pattern

Most HITL tutorials follow this flow:

```mermaid
graph TD
    A["Agent Generates Tool Call"] --> B["INTERRUPT: Ask Human"]
    B --> C{"Approve or<br/>Reject?"}
    C -->|Approve| D["Execute Tool"]
    C -->|Reject| E["End"]
    D --> E
    
    style A fill:#1e88e5,stroke:#0d47a1,stroke-width:2px,color:#fff
    style B fill:#fb8c00,stroke:#e65100,stroke-width:2px,color:#fff
    style C fill:#8e24aa,stroke:#4a148c,stroke-width:2px,color:#fff
    style D fill:#43a047,stroke:#1b5e20,stroke-width:2px,color:#fff
    style E fill:#e53935,stroke:#b71c1c,stroke-width:2px,color:#fff
```

The pattern is simple: pause the graph at a decision point, collect human input, then proceed or terminate. This works for scenarios like content moderation (should this post be deleted?) or cost estimation (do you authorize this $50 API call?).

But notice what happens if the human rejects: the conversation typically ends. If they want to retry with different parameters, they start over. There's no mechanism for the agent to learn from the rejection and self-correct.

### The Multi-Turn Edit Loop Pattern

Our implementation introduces a crucial change:

```mermaid
graph TD
    A["Agent Generates Tool Call"] --> B["INTERRUPT: Ask Human"]
    B --> C{"Approve or<br/>Reject?"}
    C -->|Reject| D["Inject Feedback<br/>as Message"]
    C -->|Approve| E["Execute Tool"]
    D --> F["Agent Processes<br/>Updated Context"]
    E --> F
    F --> A
    
    style A fill:#1e88e5,stroke:#0d47a1,stroke-width:3px,color:#fff
    style B fill:#fb8c00,stroke:#e65100,stroke-width:2px,color:#fff
    style C fill:#8e24aa,stroke:#4a148c,stroke-width:2px,color:#fff
    style D fill:#e53935,stroke:#b71c1c,stroke-width:2px,color:#fff
    style E fill:#43a047,stroke:#1b5e20,stroke-width:2px,color:#fff
    style F fill:#fbc02d,stroke:#f57f17,stroke-width:2px,color:#000
```

The key difference: rejection loops back to the agent, not to the user. The agent sees the rejection as a new human message in the conversation context and generates a revised tool call. The human approves or rejects again, creating an iterative refinement loop.

This approach has several benefits. It preserves conversation context, allows the LLM to self-correct, and creates a natural UX where users guide the agent through multiple attempts to reach the desired outcome.

---

## Architecture Overview

Let's examine the actual implementation. The graph structure in `src/agent.py` defines four nodes:

1. **agent** - Calls the LLM with tools
2. **approval** - Interrupts to request human decision
3. **tools** - Executes approved tool calls
4. **rejected** - Handles rejections by injecting feedback

The critical edges for multi-turn looping are:

- `agent` -> `approval` (conditional: if tool calls exist)
- `approval` -> `tools` (if approved)
- `approval` -> `rejected` (if rejected)
- `tools` -> `agent` (loop back)
- `rejected` -> `agent` (loop back with feedback)

Notice the two edges feeding back into `agent`: after tool execution OR after rejection. This creates the multi-turn loop.

The state schema (from `src/schemas.py`) is minimal but sufficient:

```python
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    approved: bool
    feedback: str
```

Three fields with clear responsibilities: messages track the conversation history including system, user, assistant, and tool messages; approved is a boolean flag for the approval status; and feedback holds optional user guidance for rejections. This simplicity is key. We don't need complex iteration counters or history tracking because the graph structure and message accumulation handle everything.

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

This node starts by ensuring a system message is always present at the beginning, which keeps the agent's behavior consistent across invocations. It then calls the model with the current conversation messages. The code handles an edge case where the model returns tool results but no content, which can happen when search results come back but the model hasn't summarized them yet. In that scenario, it prompts the model to provide a summary. The key point is that all responses get appended to the state via the `add_messages` handler, so the conversation history accumulates naturally with each node invocation.

### The Approval Node: Interrupting for Human Decision

The approval node is where the multi-turn loop's pause mechanism lives:

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

The `interrupt()` call from LangGraph pauses execution and sends data to the client (whether that's a CLI, web UI, or API). The client displays the tool calls and waits for human input. Once the human responds with approval status and optional feedback, that data gets captured and stored in state. This is the gateway: if approved, execution proceeds to the tools node; if rejected, execution goes to the rejection handler.

### The Rejection Handler: Closing the Loop

This is where the loop's refinement mechanism lives:

```python
def handle_rejection(state: AgentState):
    feedback = state.get("feedback", "The user rejected the tool call. Please revise your approach.")
    feedback_message = HumanMessage(content=feedback)
    
    return {"messages": [feedback_message], "approved": False}
```

It's straightforward: take the user's feedback and inject it back into the conversation as a HumanMessage. Now the agent sees its own tool call (in the previous AIMessage) plus the human's feedback (in this new HumanMessage). On the next invocation, the agent regenerates with this full context. If the feedback was "Use DuckDuckGo instead of SERP," the agent will try that. If it still doesn't work, the human rejects again, feedback accumulates, and the loop continues.

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

If the human rejected the DuckDuckGo search too, the loop would continue indefinitely, though in practice users provide better feedback or manually stop.

---

## State Management and Persistence

An important aspect of this pattern is how state persists across approvals. The answer is LangGraph's checkpoint system. In `src/agent.py`:

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

This means conversation history is preserved across approval and rejection cycles, tool calls are never executed twice because you only resume after human decision, and multiple sessions can run in parallel each with its own thread ID. For the CLI in `cli_demo.py`, a single thread ID is used per session. For Streamlit in `streamlit_demo.py`, each session gets a unique thread ID. This allows the same agent to handle multiple concurrent users.

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

The Streamlit UI in `streamlit_demo.py` handles approvals differently because it's stateful:

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

When the user clicks "Approve" or "Reject" with feedback, this function resumes the graph with the approval data. If the agent generates another tool call, the UI interrupts again and displays another approval prompt. The loop happens within Streamlit's page renders and reruns. Both implementations achieve the same multi-turn loop, just adapted to their execution model (CLI blocks synchronously; Streamlit rerenders asynchronously).

---

## Extending the Pattern for Production Use

This simple implementation can be refined for more complex production scenarios. Consider these enhancements:

**Adding iteration controls for production safety.** The current implementation allows infinite rejection cycles. For production systems handling destructive operations, you might add an explicit counter incrementing on each rejection or use timeout logic to prevent runaway loops. You could also add cost tracking to abort after too many tool invocations.

**Structured feedback for better refinement.** The multi-turn loop works best when feedback is specific and actionable. If a user rejects with just "This is wrong," the agent might struggle to improve. You could enhance this with structured feedback forms using checkboxes or dropdowns, provide examples of good feedback to guide users, or even have the agent propose corrections for the user to choose from.

**Handling stateful operations.** The current implementation only executes tools after approval, so there's no risk of repeated invocations. If you wanted to allow intermediate tool execution for debugging or real-time feedback, you would need idempotency guarantees or rollback mechanisms to protect stateful operations.

**Context management for long conversations.** Messages accumulate in state with each loop turn. For long approval cycles, this could exceed token limits. You could summarize old messages periodically, archive messages to external storage, or implement a sliding window of recent messages. The good news: when this becomes an issue, these are straightforward retrofits.

---

## Generalize Beyond Tool Approval

This pattern isn't specific to tool calls or even to search functionality. It generalizes across any workflow requiring iterative human refinement.

**Code generation workflows** use this pattern extensively. Claude Code and Cursor both implement interactive code generation where users can reject initial suggestions, provide feedback on what to change, and watch the AI regenerate improved code. The loop runs until the code meets the user's requirements.

**Content creation and editing** follows the same structure. A system generates content, the human reviews it, provides editorial notes, and the system refines. This applies to article writing, documentation, marketing copy, or any creative output.

**Data extraction and validation** benefits from multi-turn loops. Extract fields from unstructured data, show the results to a human, flag errors or inconsistencies, and have the agent re-extract with corrections. This is more reliable than hoping for accuracy on the first pass.

**API orchestration and planning** uses similar patterns. An agent plans a sequence of API calls, the human reviews the plan and flags potential issues, the agent replans with constraints, and this repeats until the plan is safe and optimal.

The architecture stays identical across these domains. You change the nodes to fit your domain (instead of search, you call a code generator; instead of rejection, you flag bad data), but the graph structure and loop mechanism remain the same. This is the power of the pattern: it's simple enough to understand quickly but flexible enough to solve diverse problems.

---

## Conclusion: Iterative Refinement as a First-Class Pattern

Single-interrupt HITL systems treat human involvement as a gating mechanism: pause, ask, proceed or abort. But real workflows are iterative. Users need to steer, adjust, and refine. With destructive operations like database deletions or API calls with side effects, iteration is non-negotiable.

The multi-turn edit loop pattern, enabled by a simple architectural change (rejection looping to agent), transforms HITL systems from binary gates into interactive refinement engines. Users don't just approve or reject; they guide the agent through multiple attempts toward the desired outcome.

Building this required a clear graph structure with explicit feedback edges, state management to track conversation history, feedback injection as conversation messages, and checkpoint persistence across approval cycles. The implementation in this codebase demonstrates that you don't need elaborate systems. A simple 4-node graph, a basic state schema, and proper message handling unlock powerful multi-turn interaction.

If you're building tool-based agents, consider this pattern early. It's a small architectural change with outsized UX benefits. Whether you're generating code, extracting data, orchestrating APIs, or searching the web, iteration is essential. Make it a first-class feature of your graph.

---

## References and Further Exploration

- **LangGraph Documentation**: https://langchain-ai.github.io/langgraph/
- **Interrupt and Command API**: Key to implementing resumable flows
- **Checkpointing**: Essential for multi-turn conversation persistence
- **Tool Binding**: How LangChain agents bind tools to language models

The code for this implementation is available in the repository. Explore `src/agent.py` for the graph definition, `src/nodes.py` for node implementations, and both `cli_demo.py` and `streamlit_demo.py` for UI-specific patterns.

Happy building.
