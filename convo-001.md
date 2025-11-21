# Conversation 001: Agentic Poker Inception

## User Requests
1.  **Core Application**: A Texas Hold'em Poker application with 6 AI agents and 1 human player.
2.  **Modes**: "Watch Mode" (Spectator) and "Play Mode" (Human vs AI).
3.  **AI Personalities**: Configurable personalities (e.g., Tight, Aggressive, Analytical) defined in a YAML file (`tune_ai_players.yml`).
4.  **Architecture**:
    *   **Agentic**: Use "Google ADK" principles (Perceive-Think-Act).
    *   **Communication**: Agent-to-Agent (A2A) using **MCP** (Model Context Protocol).
    *   **Paradigm**: **Functional Programming** (Immutable state, pure functions) over OOP.
    *   **Typing**: 100% Strongly Typed Python.
5.  **Testing**: **BDD** (Behavior Driven Development) where the Test Runner is itself an Agent.
6.  **UI**:
    *   **Tech Stack**: Streamlit with a **Custom Component** (React + TypeScript).
    *   **Aesthetics**: Premium, Futuristic, Dark Mode, Desktop-First.
    *   **Features**: Rich Info Panel (bottom 25%), Hover effects, Keyboard navigation, Toggle cards.
7.  **CLI**: A command-line runner to simulate games and log data for training.

## Actions Taken
1.  **Planning**: Created `task.md`, `implementation_plan.md`, and `code-generation-preferences.md`.
2.  **Domain Modeling**: Defined immutable data structures (`Card`, `Player`, `GameState`) in `engine/domain.py`.
3.  **Core Logic**: Implemented pure game functions (`apply_action`, `next_player_index`) in `engine/logic.py`.
4.  **Protocol Design**: Defined MCP message types (`ActionRequest`, `GameStateUpdate`) in `engine/mcp.py`.
5.  **Agent Implementation**: Created `FunctionalAgent` in `agents/functional_agent.py` that loads personalities from YAML.
6.  **Testing**: Built a `BDDAgentRunner` (`tests/agent_runner.py`) and Gherkin features (`poker_rules.feature`) to verify logic.
7.  **UI Development**:
    *   Scaffolded a React+Vite project inside `ui/poker_component`.
    *   Built `Table`, `Player`, and `InfoPanel` components with Tailwind CSS.
    *   Integrated into Streamlit via `ui/app.py`.
8.  **Simulation**: Implemented `cli.py` to run high-speed simulations and generate JSONL logs.

## High-Level Approach
The system is designed as a **Functional Multi-Agent System**.
*   **The World**: The `Dealer` (or Game Engine) manages the immutable `GameState`. It is the source of truth.
*   **The Actors**: Agents are stateless functions (conceptually) that take a `GameState` and a `Personality` and output an `Action`.
*   **The Glue**: **MCP** messages carry state and requests between the World and the Actors.

## Design Decisions
1.  **Functional Core**: We chose immutable dataclasses (`frozen=True`) to prevent side effects and make the game state easy to debug and replay (Time Travel debugging potential).
2.  **MCP for A2A**: Instead of direct method calls, agents communicate via typed messages. This decouples the agents from the engine and allows for future expansion (e.g., running agents in separate containers).
3.  **React for UI**: Standard Streamlit widgets were insufficient for the "Premium" and "High Interactivity" requirements (hover states, animations). We injected a React app to handle the game board while letting Streamlit handle the outer shell.
4.  **BDD as an Agent**: We treated the Test Runner as a "God Agent" that can inject state and assert outcomes, aligning the testing strategy with the agentic architecture.
5.  **Desktop-First UI**: We optimized the layout for 16:9 screens, using a high-density dashboard in the bottom 25% to show advanced analytics, catering to the "Analytical" user persona.
