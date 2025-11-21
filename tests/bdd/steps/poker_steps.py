from behave import given, when, then
from tests.agent_runner import BDDAgentRunner


@given('a new game with players "{player_names}"')
def step_impl(context, player_names):
    names = [n.strip() for n in player_names.split(",")]
    context.runner = BDDAgentRunner()
    context.runner.setup_game(names)


@given('"{player_name}" has cards "{cards}"')
def step_cards(context, player_name, cards):
    context.runner.inject_cards(player_name, cards)


@given('the community cards are "{cards}"')
def step_community_cards(context, cards):
    context.runner.inject_community_cards(cards)


@when('"{player_name}" checks')
def step_check(context, player_name):
    context.runner.player_acts(player_name, "CHECK")


@when('"{player_name}" raises to {amount:d}')
def step_raise(context, player_name, amount):
    context.runner.player_acts(player_name, "RAISE", amount)


@then("the pot should be {amount:d}")
def step_pot_check(context, amount):
    assert context.runner.game_state.pot.amount == amount


@then('the phase should be "{phase_name}"')
def step_phase_check(context, phase_name):
    current_phase = context.runner.game_state.phase.name
    assert current_phase == phase_name, f"Expected phase {phase_name}, but got {current_phase}"


@then('"{player_name}" should be the winner')
def step_winner_check(context, player_name):
    # Check history or chips to determine winner
    # In our logic, winner gets chips and history says "Winners: ..."
    history = context.runner.game_state.history
    winner_log = next((h for h in reversed(history) if "Winners:" in h), None)
    assert winner_log is not None, "No winner declared in history"
    assert player_name in winner_log, f"Expected {player_name} to win, but log says: {winner_log}"
