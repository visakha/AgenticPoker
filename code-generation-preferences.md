## Architecture & Agents
- **Agent Framework**: Use **Google ADK** (Agent Development Kit) principles.
- **Communication**: Implement **A2A** (Agent-to-Agent) communication.
- **Protocol**: Use **MCP** (Model Context Protocol) for standardized context exchange between agents.

## Testing (BDD)
- **Framework**: Generate BDD test cases (Gherkin syntax).
- **Runner**: The BDD Test Runner must itself be implemented as an **Agent**.
  - It sets up the environment (Dealer).
  - It invokes Player Agents.
  - It observes the game state to verify scenarios.
- **Documentation**: Maintain a `dev-bdd-guide.md` explaining this architecture.

## UI
- **Tech Stack**: Streamlit Custom Component with **React** + **TypeScript**.
- **Style**: No vanilla JS for logic; use TypeScript.

## Tone & Style
- **Communication**: Be funny, cheerful, and casual. This is a hobby project!
- **Documentation**: Use emojis, jokes, and keep it lighthearted.

## Python
- **Version**: Use Python 3.13 or newer.
- **Libraries**: Use the latest stable versions of libraries.
- **Code Style**: Use **Black** for code formatting.
- **Type Hints**: Use **Type Hints** for type safety.
- **Documentation**: Use **Sphinx** for documentation.
- **Functional Programming**: adopt functional programming principles. minimize OOP
