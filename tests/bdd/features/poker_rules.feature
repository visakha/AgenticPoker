Feature: Basic Poker Rules

  Scenario: Players can check and raise
    Given a new game with players "Alice, Bob"
    And "Alice" has cards "Ah Kh"
    When "Alice" raises to 100
    Then the pot should be 100 
    # Pot is 0 because chips aren't collected until round end in some engines, 
    # or immediately. We need to define this in logic.py. 
    # For now, let's assume our logic updates player chips but maybe not pot yet.

  Scenario: Players check around to Showdown
    Given a new game with players "Alice, Bob"
    And "Alice" has cards "Ah Kh"
    And "Bob" has cards "2c 7d"
    And the community cards are "Ad Kd Qd Jd Td"
    When "Alice" checks
    And "Bob" checks
    Then the phase should be "FLOP"
    When "Alice" checks
    And "Bob" checks
    Then the phase should be "TURN"
    When "Alice" checks
    And "Bob" checks
    Then the phase should be "RIVER"
    When "Alice" checks
    And "Bob" checks
    Then the phase should be "SHOWDOWN"
    And "Alice" should be the winner
