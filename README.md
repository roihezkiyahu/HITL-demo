# Human-in-the-Loop Agent Demo

A simple demonstration of a Human-in-the-Loop (HITL) agent using LangGraph with Google Gemini and web search capability. This demo showcases repetitive editing through an interrupt-based approval flow.

## Architecture

The agent is built using LangGraph's StateGraph with the following flow:

```
agent → approval → tools → agent
         ↓
      rejected → agent
```

- **agent**: Calls the LLM to generate responses
- **approval**: Interrupts execution to request human approval for tool calls
- **tools**: Executes approved tool calls
- **rejected**: Handles rejection and injects feedback back to the agent

## Features

- ✅ Human approval required before tool execution
- ✅ Repetitive editing via rejection feedback
- ✅ Conversation memory across multiple turns
- ✅ Both CLI and Streamlit interfaces
- ✅ Simple, blog-friendly code structure

## Prerequisites

- Python 3.10 or higher
- Google API key for Gemini access

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables:**
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your Google API key:
   ```
   GOOGLE_API_KEY=your_actual_api_key_here
   ```

## Usage

### CLI Demo

Run the command-line interface:

```bash
python cli_demo.py
```

**Example interaction:**
```
You: What's the weather in Paris?

TOOL CALL APPROVAL REQUIRED
Tool Call #1:
  Tool: search_web
  Arguments: {'query': 'weather in Paris'}

Approve these tool calls? (yes/no): no
Provide feedback: Search for weather in London instead

[Agent revises and proposes new search]
...
```

### Streamlit Demo

Run the web interface:

```bash
streamlit run streamlit_demo.py
```

Then open your browser to the displayed URL (typically http://localhost:8501).

**Features:**
- Chat-style interface
- Visual tool call approval with expandable details
- Rejection feedback text area
- Clear chat history button

## How It Works

### Repetitive Editing

The key feature of this demo is the ability to repeatedly reject and edit the agent's actions:

1. Agent proposes a tool call
2. Human can **approve** or **reject**
3. If rejected, human provides feedback
4. Agent receives feedback and revises approach
5. Process repeats until human approves

This creates a collaborative loop where the human guides the agent to the desired outcome.

### State Management

The agent maintains state across turns using LangGraph's `MemorySaver` checkpointer. Each conversation has a unique `thread_id` that preserves:
- Message history
- Previous tool calls
- Context for follow-up questions

## File Structure

```
demo/
├── agent.py              # Agent initialization with StateGraph
├── nodes.py              # Node functions (agent, approval, tools, rejection)
├── schemas.py            # AgentState TypedDict
├── tools.py              # Web search tool
├── prompts.py            # System prompt
├── cli_demo.py           # CLI interface
├── streamlit_demo.py     # Streamlit web interface
├── requirements.txt      # Python dependencies
├── .env.example          # Environment variable template
└── README.md             # This file
```

## Code Simplicity

This demo is designed for a Medium blog post with these principles:
- **Minimal abstractions**: Direct, readable code
- **Clear separation**: Each file has a single purpose
- **No over-engineering**: Only essential features
- **Well-commented**: Explains what and why

## Example Use Cases

1. **Weather queries**: "What's the weather in Tokyo?"
2. **Current events**: "What's the latest news about AI?"
3. **Research**: "Find information about LangGraph"
4. **Fact-checking**: "When did the latest SpaceX launch happen?"

## Troubleshooting

**Issue**: `GOOGLE_API_KEY not found`
- **Solution**: Make sure you've created a `.env` file and added your API key

**Issue**: `Module not found` errors
- **Solution**: Run `pip install -r requirements.txt`

**Issue**: DuckDuckGo search fails
- **Solution**: Check your internet connection; DuckDuckGo search may occasionally be rate-limited

## License

This demo is provided as-is for educational purposes.

