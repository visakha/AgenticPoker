Agentic Poker Documentation
============================

Welcome to the **Agentic Poker** documentation! 🃏🤖

This is a Texas Hold'em poker engine where autonomous AI agents battle it out using the **Google ADK** (Agent Development Kit) principles and **MCP** (Model Context Protocol) for communication.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   architecture
   api/index
   bdd_testing
   contributing

Quick Start
-----------

Install dependencies::

    pip install -r requirements.txt
    cd ui/poker_component && npm install && npm run build

Run the application::

    streamlit run ui/app.py

Architecture Overview
--------------------

The system follows an **Agentic Architecture**:

* **Dealer Agent**: Manages game state and orchestrates player actions
* **Player Agents**: Make decisions based on personality and game state
* **MCP Protocol**: Standardized message passing between agents
* **Functional Core**: Immutable data structures and pure functions

Key Features
-----------

* 🎮 **Watch Mode**: Observe AI agents playing poker
* 🎲 **Play Mode**: Compete against AI personalities
* 🧪 **BDD Testing**: Behavior-driven development with Gherkin scenarios
* 📊 **Analytics**: Post-game analysis of agent behavior (VPIP, PFR, Aggression)
* 🎨 **Rich UI**: React + TypeScript custom Streamlit component

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
