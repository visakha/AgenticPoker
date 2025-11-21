Feature: Advanced Poker Scenarios

  Scenario: Player goes all-in with insufficient chips
    Given a new game with players "Alice, Bob"
    And "Alice" has 50 chips
    And "Bob" has 1000 chips
    When "Alice" raises to 1000
    Then "Alice" should be "ALL_IN"
    And "Alice" should have 0 chips
    And the pot should be 50

  Scenario: Multiple players fold
    Given a new game with players "Alice, Bob, Charlie"
    When "Alice" folds
    And "Bob" folds
    Then "Charlie" should be the winner

  Scenario: Tie with identical hands
    Given a new game with players "Alice, Bob"
    And "Alice" has cards "Ah Kh"
    And "Bob" has cards "Ad Kd"
    When the community cards are "Qh Qd Qc Jh Th"
    And the game advances to "SHOWDOWN"
    Then the pot should be split between "Alice, Bob"

  Scenario: Side pot creation
    Given a new game with players "Alice, Bob, Charlie"
    And "Alice" has 100 chips
    And "Bob" has 500 chips
    And "Charlie" has 500 chips
    When "Alice" raises to 100
    And "Bob" raises to 500
    And "Charlie" calls
    Then there should be a main pot of 300
    And there should be a side pot of 800

  Scenario: Straight beats three of a kind
    Given a new game with players "Alice, Bob"
    And "Alice" has cards "5h 6h"
    And "Bob" has cards "Kc Kd"
    When the community cards are "7s 8d 9c Kh 2s"
    And the game advances to "SHOWDOWN"
    Then "Alice" should be the winner

  Scenario: Flush beats straight
    Given a new game with players "Alice, Bob"
    And "Alice" has cards "Ah 2h"
    And "Bob" has cards "5c 6c"
    When the community cards are "3h 7h 9h 4d 8s"
    And the game advances to "SHOWDOWN"
    Then "Alice" should be the winner
