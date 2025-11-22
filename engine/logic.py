import random
from typing import List, Tuple, Optional, Dict
from .domain import (
    Card,
    Rank,
    Suit,
    GameState,
    Player,
    PlayerID,
    HandPhase,
    ActionType,
    ChipCount,
    PlayerStatus,
    Pot,
)
from .hand_evaluator import evaluate_hand

# --- Deck Functions ---


def create_deck() -> List[Card]:
    return [Card(rank, suit) for suit in Suit for rank in Rank]


def shuffle_deck(deck: List[Card], seed: int) -> List[Card]:
    rng = random.Random(seed)
    shuffled = deck[:]
    rng.shuffle(shuffled)
    return shuffled


def draw_cards(deck: List[Card], count: int) -> Tuple[List[Card], List[Card]]:
    """Returns (drawn_cards, remaining_deck)"""
    return deck[:count], deck[count:]


# --- Core Game Logic ---


def apply_action(
    state: GameState, player_id: PlayerID, action_type: ActionType, amount: ChipCount = ChipCount(0)
) -> GameState:
    """
    Pure function to apply an action to the game state.
    Returns a NEW GameState object.
    """
    # 1. Validate Action (Simplified)
    player = state.get_player(player_id)
    if not player or player.status != PlayerStatus.ACTIVE:
        return state  # Ignore invalid

    # 2. Update Player State
    new_players = []
    pot_contribution = 0

    for p in state.players:
        if p.id == player_id:
            new_chips = p.chips
            new_bet = p.current_bet
            new_status = p.status

            if action_type == ActionType.FOLD:
                new_status = PlayerStatus.FOLDED
            elif action_type in (ActionType.CALL, ActionType.CHECK):
                # Check is just a call of 0 if bets match
                to_call = state.current_bet - p.current_bet
                if to_call > 0:
                    contribution = min(to_call, p.chips)
                    new_chips = ChipCount(p.chips - contribution)
                    new_bet = ChipCount(p.current_bet + contribution)
                    pot_contribution = contribution
            elif action_type == ActionType.RAISE:
                # Raise to 'amount' (total bet)
                contribution = amount - p.current_bet
                if contribution > p.chips:  # All-in logic if raise > chips
                    contribution = p.chips
                    amount = ChipCount(p.current_bet + contribution)

                new_chips = ChipCount(p.chips - contribution)
                new_bet = amount
                pot_contribution = contribution

            if new_chips == 0 and new_status == PlayerStatus.ACTIVE:
                new_status = PlayerStatus.ALL_IN

            updated_player = Player(
                id=p.id,
                name=p.name,
                chips=new_chips,
                status=new_status,
                hole_cards=p.hole_cards,
                current_bet=new_bet,
                is_human=p.is_human,
                personality=p.personality,
                acted_this_round=True,  # Mark as acted
            )
            new_players.append(updated_player)
        else:
            new_players.append(p)

    # 3. Update Pot
    new_pot_amount = ChipCount(state.pot.amount + pot_contribution)
    new_pot = Pot(new_pot_amount, state.pot.eligible_players)

    # Update Global Bet
    new_global_bet = state.current_bet
    if action_type == ActionType.RAISE:
        new_global_bet = amount
        # Reset acted_this_round for others?
        # In poker, if someone raises, everyone else must act again.
        # So we should reset acted_this_round for all OTHER active players.
        # But this is a pure function, we need to iterate again or do it in one pass?
        # Doing it in a second pass for clarity.

        reset_acted_players = []
        for p in new_players:
            if p.id != player_id and p.status == PlayerStatus.ACTIVE:
                reset_acted_players.append(
                    Player(
                        id=p.id,
                        name=p.name,
                        chips=p.chips,
                        status=p.status,
                        hole_cards=p.hole_cards,
                        current_bet=p.current_bet,
                        is_human=p.is_human,
                        personality=p.personality,
                        acted_this_round=False,  # Reset
                    )
                )
            else:
                reset_acted_players.append(p)
        new_players = reset_acted_players

    # Intermediate State
    temp_state = GameState(
        id=state.id,
        phase=state.phase,
        community_cards=state.community_cards,
        pot=new_pot,
        current_bet=new_global_bet,
        players=new_players,
        dealer_index=state.dealer_index,
        current_player_index=state.current_player_index,
        deck_seed=state.deck_seed,
        history=state.history
        + [f"{player.name} {action_type.name} {amount if action_type==ActionType.RAISE else ''}"],
    )

    # 4. Check Phase Transition
    if _is_betting_round_over(temp_state):
        return _advance_phase(temp_state)
    else:
        return _advance_player(temp_state)


def _is_betting_round_over(state: GameState) -> bool:
    active_players = [p for p in state.players if p.status == PlayerStatus.ACTIVE]
    all_in_players = [p for p in state.players if p.status == PlayerStatus.ALL_IN]

    # If only one player remains (all others folded), hand is over
    if len(active_players) + len(all_in_players) < 2:
        return True

    # If only one active player and rest are all-in, go to showdown
    if len(active_players) == 1 and len(all_in_players) > 0:
        return True

    # Round is over if:
    # 1. All active players have acted this round.
    # 2. All active players have matched the current bet.

    all_acted = all(p.acted_this_round for p in active_players)
    all_matched = all(p.current_bet == state.current_bet for p in active_players)

    return all_acted and all_matched


def _advance_player(state: GameState) -> GameState:
    next_idx = next_player_index(state)
    return GameState(
        id=state.id,
        phase=state.phase,
        community_cards=state.community_cards,
        pot=state.pot,
        current_bet=state.current_bet,
        players=state.players,
        dealer_index=state.dealer_index,
        current_player_index=next_idx,
        deck_seed=state.deck_seed,
        history=state.history,
    )


def next_player_index(state: GameState) -> int:
    num_players = len(state.players)
    current = state.current_player_index
    for i in range(1, num_players + 1):
        idx = (current + i) % num_players
        player = state.players[idx]
        if player.status == PlayerStatus.ACTIVE:
            return idx
    return -1


def _advance_phase(state: GameState) -> GameState:
    """Advance to the next phase or resolve the hand."""
    
    # Check if only one player remains (all others folded)
    active_players = [p for p in state.players if p.status == PlayerStatus.ACTIVE]
    all_in_players = [p for p in state.players if p.status == PlayerStatus.ALL_IN]
    
    # If only one active player remains, they win immediately
    if len(active_players) == 1 and len(all_in_players) == 0:
        winner = active_players[0]
        return _award_pot_to_winner(state, winner, "All others folded")
    
    # If only one active and some all-in, go to showdown
    if len(active_players) <= 1 and len(all_in_players) > 0:
        # Reset bets and go to showdown
        reset_players = [
            Player(
                id=p.id,
                name=p.name,
                chips=p.chips,
                status=p.status,
                hole_cards=p.hole_cards,
                current_bet=ChipCount(0),
                is_human=p.is_human,
                personality=p.personality,
                acted_this_round=False,
            )
            for p in state.players
        ]
        return _resolve_showdown(state, reset_players)
    
    # 1. Reset bets and acted status for next betting round
    reset_players = [
        Player(
            id=p.id,
            name=p.name,
            chips=p.chips,
            status=p.status,
            hole_cards=p.hole_cards,
            current_bet=ChipCount(0),
            is_human=p.is_human,
            personality=p.personality,
            acted_this_round=False,
        )  # Reset for new round
        for p in state.players
    ]

    new_phase = state.phase
    new_community = state.community_cards

    # Mock dealing logic
    deck = shuffle_deck(create_deck(), state.deck_seed + len(state.history))
    used_cards = set(state.community_cards)
    for p in state.players:
        used_cards.update(p.hole_cards)
    available_deck = [c for c in deck if c not in used_cards]

    if state.phase == HandPhase.PREFLOP:
        new_phase = HandPhase.FLOP
        deal, _ = draw_cards(available_deck, 3)
        new_community = state.community_cards + deal
    elif state.phase == HandPhase.FLOP:
        new_phase = HandPhase.TURN
        deal, _ = draw_cards(available_deck, 1)
        new_community = state.community_cards + deal
    elif state.phase == HandPhase.TURN:
        new_phase = HandPhase.RIVER
        deal, _ = draw_cards(available_deck, 1)
        new_community = state.community_cards + deal
    elif state.phase == HandPhase.RIVER:
        new_phase = HandPhase.SHOWDOWN
        return _resolve_showdown(state, reset_players)

    # Find first player after dealer
    temp_state = GameState(
        players=reset_players,
        current_player_index=state.dealer_index,
        dealer_index=state.dealer_index,
    )
    next_active = next_player_index(temp_state)

    return GameState(
        id=state.id,
        phase=new_phase,
        community_cards=new_community,
        pot=state.pot,
        current_bet=ChipCount(0),
        players=reset_players,
        dealer_index=state.dealer_index,
        current_player_index=next_active,
        deck_seed=state.deck_seed,
        history=state.history + [f"--- {new_phase.name} ---"],
    )


def _award_pot_to_winner(state: GameState, winner: Player, reason: str) -> GameState:
    """Award the entire pot to a single winner."""
    final_players = []
    for p in state.players:
        if p.id == winner.id:
            final_players.append(
                Player(
                    id=p.id,
                    name=p.name,
                    chips=ChipCount(p.chips + state.pot.amount),
                    status=p.status,
                    hole_cards=p.hole_cards,
                    current_bet=ChipCount(0),
                    is_human=p.is_human,
                    personality=p.personality,
                    acted_this_round=False,
                )
            )
        else:
            final_players.append(
                Player(
                    id=p.id,
                    name=p.name,
                    chips=p.chips,
                    status=p.status,
                    hole_cards=p.hole_cards,
                    current_bet=ChipCount(0),
                    is_human=p.is_human,
                    personality=p.personality,
                    acted_this_round=False,
                )
            )

    return GameState(
        id=state.id,
        phase=HandPhase.SHOWDOWN,
        community_cards=state.community_cards,
        pot=Pot(ChipCount(0), []),
        current_bet=ChipCount(0),
        players=final_players,
        dealer_index=state.dealer_index,
        current_player_index=-1,
        deck_seed=state.deck_seed,
        history=state.history + [f"Winner: {winner.name} ({reason}) - Won ${state.pot.amount}"],
    )


def _resolve_showdown(state: GameState, players: List[Player]) -> GameState:
    active_players = [p for p in players if p.status in (PlayerStatus.ACTIVE, PlayerStatus.ALL_IN)]

    if not active_players:
        return state

    winners = []
    best_rank = -1
    best_kickers = []

    for p in active_players:
        rank, kickers = evaluate_hand(p.hole_cards, state.community_cards)
        if rank > best_rank:
            best_rank = rank
            best_kickers = kickers
            winners = [p]
        elif rank == best_rank:
            if kickers > best_kickers:
                best_kickers = kickers
                winners = [p]
            elif kickers == best_kickers:
                winners.append(p)

    share = state.pot.amount // len(winners)
    winner_names = [w.name for w in winners]

    final_players = []
    for p in players:
        if p in winners:
            final_players.append(
                Player(
                    id=p.id,
                    name=p.name,
                    chips=ChipCount(p.chips + share),
                    status=p.status,
                    hole_cards=p.hole_cards,
                    current_bet=p.current_bet,
                    is_human=p.is_human,
                    personality=p.personality,
                    acted_this_round=False,
                )
            )
        else:
            final_players.append(p)

    return GameState(
        id=state.id,
        phase=HandPhase.SHOWDOWN,
        community_cards=state.community_cards,
        pot=Pot(ChipCount(0), []),
        current_bet=ChipCount(0),
        players=final_players,
        dealer_index=state.dealer_index,
        current_player_index=-1,
        deck_seed=state.deck_seed,
        history=state.history + [f"Winners: {', '.join(winner_names)} ({best_rank.name})"],
    )
