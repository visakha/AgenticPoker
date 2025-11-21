Architecture
============

Agentic Design Pattern
----------------------

**Agentic Poker** uses an agent-based architecture where each component is an autonomous agent that:

1. **Perceives** the game state
2. **Thinks** about the best action
3. **Acts** by sending messages

Agents
------

Dealer Agent
~~~~~~~~~~~~

The Dealer is the orchestrator. It:

* Manages the immutable ``GameState``
* Deals cards and advances game phases
* Sends ``GAME_STATE_UPDATE`` messages to players
* Requests actions via ``ACTION_REQUEST`` messages
* Applies player actions using pure functions

Player Agents
~~~~~~~~~~~~~

Each player is an autonomous agent with:

* **Personality**: Loaded from ``tune_ai_players.yml`` (VPIP, PFR, Aggression, Bluff Frequency)
* **Decision Logic**: Functional algorithm in ``_decide_action()``
* **Dialogue**: Context-aware trash talk via ``DialogueManager``

MCP (Model Context Protocol)
-----------------------------

All agent communication uses typed MCP messages:

.. code-block:: python

    @dataclass(frozen=True)
    class MCPMessage:
        sender: str
        recipient: str
        message_type: str
        payload: Any

Message Types
~~~~~~~~~~~~~

* ``GAME_STATE_UPDATE``: Dealer → Player (state snapshot)
* ``ACTION_REQUEST``: Dealer → Player (request move)
* ``ACTION_RESPONSE``: Player → Dealer (chosen action)
* ``DIALOGUE_EVENT``: Any → All (chat/trash talk)

Functional Core
---------------

The game logic is **purely functional**:

* **Immutable Data**: All ``@dataclass(frozen=True)``
* **Pure Functions**: ``apply_action()`` returns new ``GameState``
* **No Side Effects**: State transitions are deterministic

Example::

    new_state = apply_action(state, player_id, ActionType.RAISE, ChipCount(100))

This makes the system:

* **Testable**: Easy to write BDD scenarios
* **Debuggable**: State history is traceable
* **Parallelizable**: No shared mutable state
