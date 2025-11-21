# BDD Agent Runner Guide

## Overview
The **BDD Agent Runner** (`tests/agent_runner.py`) is a specialized agent designed to test the Agentic Poker system. Unlike standard unit tests, it simulates the **Agentic Environment**.

## Architecture
1.  **Gherkin Features**: Define scenarios in plain English (e.g., `poker_rules.feature`).
2.  **Step Definitions**: Map Gherkin steps to Python code.
3.  **BDD Agent**:
    *   Maintains the `GameState`.
    *   Injects state (cards, chips) using "God Mode" privileges.
    *   Simulates Agent Actions (MCP Messages).
    *   Verifies the resulting state.

## How to Run
```bash
behave tests/bdd/features/
```

## Extending
To add new steps, edit `tests/agent_runner.py` and add `@given`, `@when`, `@then` decorators.
