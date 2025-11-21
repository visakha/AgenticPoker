import json
import argparse
from collections import defaultdict
from typing import Dict, List


class StatsAnalyzer:
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.stats = defaultdict(
            lambda: {
                "hands": 0,
                "vpip_count": 0,
                "pfr_count": 0,
                "bet_raise_count": 0,
                "call_check_count": 0,
            }
        )

    def analyze(self):
        print(f"Analyzing {self.log_file}...")
        try:
            with open(self.log_file, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    game_data = json.loads(line)
                    self._process_game(game_data)

            self._print_report()
        except FileNotFoundError:
            print(f"Error: File {self.log_file} not found.")

    def _process_game(self, game_data: Dict):
        players = game_data.get("players", [])
        actions = game_data.get("actions", [])

        # Track who played in this hand
        active_players = set()

        # Track actions per player for this hand
        player_actions = defaultdict(list)

        for action in actions:
            p_name = action["player"]
            act_type = action["action"]
            phase = action["phase"]

            active_players.add(p_name)
            player_actions[p_name].append((act_type, phase))

            # Aggression stats (Post-flop only usually, but let's do total for MVP)
            if act_type in ["BET", "RAISE", "ALL_IN"]:
                self.stats[p_name]["bet_raise_count"] += 1
            elif act_type in ["CALL", "CHECK"]:
                self.stats[p_name]["call_check_count"] += 1

        for p_name in players:
            self.stats[p_name]["hands"] += 1

            # VPIP: Voluntarily Put Money In Pot (Call or Raise preflop)
            # Simplified: Did they do anything other than Fold or Check Preflop?
            # Actually, Check is forced BB or free.
            # Let's look for CALL or RAISE in PREFLOP

            actions_list = player_actions[p_name]
            preflop_actions = [a[0] for a in actions_list if a[1] == "PREFLOP"]

            if (
                "CALL" in preflop_actions
                or "RAISE" in preflop_actions
                or "ALL_IN" in preflop_actions
            ):
                self.stats[p_name]["vpip_count"] += 1

            # PFR: Pre-Flop Raise
            if "RAISE" in preflop_actions or "ALL_IN" in preflop_actions:
                self.stats[p_name]["pfr_count"] += 1

    def _print_report(self):
        print("\n" + "=" * 60)
        print(f"{'PLAYER':<20} | {'HANDS':<5} | {'VPIP':<6} | {'PFR':<6} | {'AGG':<6}")
        print("-" * 60)

        for p_name, data in self.stats.items():
            hands = data["hands"]
            if hands == 0:
                continue

            vpip = (data["vpip_count"] / hands) * 100
            pfr = (data["pfr_count"] / hands) * 100

            calls_checks = data["call_check_count"]
            if calls_checks > 0:
                agg = data["bet_raise_count"] / calls_checks
            else:
                agg = data["bet_raise_count"]  # Infinite? Just show raw count

            print(f"{p_name:<20} | {hands:<5} | {vpip:>5.1f}% | {pfr:>5.1f}% | {agg:>5.2f}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agentic Poker Stats Analyzer")
    parser.add_argument("logfile", help="Path to JSONL log file")
    args = parser.parse_args()

    analyzer = StatsAnalyzer(args.logfile)
    analyzer.analyze()
