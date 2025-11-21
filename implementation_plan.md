# Agentic Poker Implementation Plan

## Goal Description
Build a Texas Hold'em poker application with a Streamlit UI and a CLI runner. The system will be a **Multi-Agent System (MAS)** using **MCP** and **A2A** communication. The codebase will adhere to **Functional Programming** principles and be **100% strongly typed**.

## User Review Required
> [!IMPORTANT]
> **Architectural Overhaul**:
> - **Functional Core**: Game logic will be pure functions. State will be immutable `dataclasses`.
> - **MCP & A2A**: Agents will communicate via a typed message bus implementing the Model Context Protocol.
> - **BDD Agent**: Testing will be driven by a "Tester Agent" that orchestrates scenarios.

## Proposed Changes

### Project Structure
```
AgenticPoker/
├── agents/
│   ├── __init__.py
│   ├── functional_agent.py # [NEW] Functional agent wrapper
│   ├── personalities.py
│   └── dialogue.py
├── engine/
│   ├── __init__.py
│   ├── domain.py           # [NEW] Immutable Data Structures (Card, State)
│   ├── logic.py            # [NEW] Pure Functions (eval_hand, next_state)
│   └── mcp.py              # [NEW] Model Context Protocol definitions
├── tests/
│   ├── bdd/
│   │   ├── features/       # Gherkin files
│   │   └── steps/
│   └── agent_runner.py     # [NEW] The BDD Agent
├── ui/
│   ├── app.py
│   └── poker_component/    # React + TypeScript
├── config/
│   └── tune_ai_players.yml
├── docs/
│   ├── user_guide.md
│   └── dev-bdd-guide.md    # [NEW]
└── main.py
```

### Core Engine (Functional)
#### [NEW] [engine/domain.py](file:///C:/Users/vamsi/wrk-py/AgenticPoker/engine/domain.py)
- Immutable `FrozenDataclass` for `Card`, `Player`, `GameState`.
- **Strict Typing**: usage of `NewType`, `Literal`, `Sequence`.

#### [NEW] [engine/logic.py](file:///C:/Users/vamsi/wrk-py/AgenticPoker/engine/logic.py)
- Pure functions: `(GameState, Action) -> GameState`.
- No side effects.

### Agents & MCP
#### [NEW] [engine/mcp.py](file:///C:/Users/vamsi/wrk-py/AgenticPoker/engine/mcp.py)
- Defines the schema for A2A messages (Context, Resources, Prompts).
- Implements the "Protocol" for agent negotiation.

### BDD Testing
#### [NEW] [tests/agent_runner.py](file:///C:/Users/vamsi/wrk-py/AgenticPoker/tests/agent_runner.py)
- A special Agent that parses Gherkin features.
- Injects specific game states (Context) via MCP.
- Verifies agent responses.

#### [NEW] [docs/dev-bdd-guide.md](file:///C:/Users/vamsi/wrk-py/AgenticPoker/docs/dev-bdd-guide.md)
- Documentation on how the BDD Agent works and how to write new tests.

## Verification Plan

### Automated Tests
- **BDD**: Run `behave` or custom runner using `tests/agent_runner.py`.
- **Type Check**: `mypy --strict .` (Must pass 100%).

### Manual Verification
- **UI**: Verify React component renders state correctly from the functional core.
