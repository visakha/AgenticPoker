BDD Testing
===========

The **BDD Agent Runner** is a unique testing approach where the test framework itself is an autonomous agent.

Philosophy
----------

Traditional unit tests check individual functions. BDD tests verify **behavior** from a user's perspective using plain English (Gherkin syntax).

In Agentic Poker, the **Tester Agent** acts as a "God Mode" observer that:

1. Sets up game scenarios
2. Injects specific cards/state
3. Simulates player actions
4. Verifies outcomes

Writing Tests
-------------

Tests are written in Gherkin (``tests/bdd/features/*.feature``):

.. code-block:: gherkin

    Feature: Basic Poker Rules

      Scenario: Player wins with better hand
        Given a new game with players "Alice, Bob"
        And "Alice" has cards "Ah Kh"
        And "Bob" has cards "2c 3d"
        When the community cards are "Ad Kd Qh Jh Th"
        And the game advances to "SHOWDOWN"
        Then "Alice" should be the winner

Step Definitions
----------------

Step definitions map Gherkin to Python (``tests/bdd/steps/poker_steps.py``):

.. code-block:: python

    @given('a new game with players "{players}"')
    def step_new_game(context, players):
        context.runner = BDDAgentRunner()
        context.runner.setup_game(players.split(", "))

The BDD Agent Runner
--------------------

Located in ``tests/agent_runner.py``, the runner provides:

* ``setup_game(players)``: Initialize game state
* ``inject_cards(player, cards)``: Give specific hole cards
* ``inject_community_cards(cards)``: Set the board
* ``simulate_action(player, action, amount)``: Force an action
* ``get_state()``: Inspect current game state

Running Tests
-------------

Run all BDD tests::

    behave tests/bdd/features/

Run specific feature::

    behave tests/bdd/features/poker_rules.feature

Run with verbose output::

    behave -v tests/bdd/features/

Adding New Scenarios
--------------------

1. Create a ``.feature`` file in ``tests/bdd/features/``
2. Write scenarios in Gherkin
3. Add step definitions in ``tests/bdd/steps/``
4. Run ``behave`` to verify

Example: Testing All-In
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: gherkin

    Scenario: Player goes all-in
      Given a new game with players "Alice, Bob"
      And "Alice" has 100 chips
      When "Alice" raises to 100
      Then "Alice" should be "ALL_IN"
      And the pot should be 100
