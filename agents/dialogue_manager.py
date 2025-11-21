import random
from typing import List, Optional
from engine.domain import GameState, Player, ActionType


class DialogueManager:
    """
    Manages agent dialogue generation.
    In a real implementation, this would call an LLM (Gemini/GPT).
    For this prototype, we use template-based responses.
    """

    def __init__(self, use_llm: bool = False):
        self.use_llm = use_llm

    def generate_dialogue(
        self, player: Player, action: ActionType, state: GameState
    ) -> Optional[str]:
        """
        Generates a line of dialogue based on the player's personality and action.
        Returns None if the agent decides to stay silent.
        """
        # 30% chance to speak
        if random.random() > 0.3:
            return None

        personality = player.personality.lower()

        if self.use_llm:
            return self._call_llm(player, action, state)
        else:
            return self._template_response(player, action, personality)

    def _template_response(self, player: Player, action: ActionType, personality: str) -> str:
        templates = {
            "tight": {
                ActionType.FOLD: [
                    "Too rich for my blood.",
                    "I'll wait for a better spot.",
                    "Fold.",
                ],
                ActionType.RAISE: ["I have the nuts.", "You should be scared.", "Raising."],
                ActionType.CHECK: ["Check.", "Checking.", "No bet."],
                ActionType.CALL: ["I'll keep you honest.", "Calling.", "Let's see the next card."],
            },
            "aggressive": {
                ActionType.FOLD: ["This is boring.", "Whatever.", "Next hand."],
                ActionType.RAISE: ["PAY ME!", "I'm all in... almost.", "Try to catch me!"],
                ActionType.CHECK: ["Trapping...", "Check.", "Go ahead, bet."],
                ActionType.CALL: ["You're bluffing.", "I'm not going anywhere.", "Call."],
            },
            "analytical": {
                ActionType.FOLD: ["Pot odds aren't there.", "Negative EV.", "Folding range."],
                ActionType.RAISE: [
                    "Maximizing value.",
                    "Exploiting your weakness.",
                    "Calculating... Raise.",
                ],
                ActionType.CHECK: ["Variance is high.", "Check.", "Standard line."],
                ActionType.CALL: ["Price is right.", "Float.", "Calling station mode."],
            },
        }

        # Fallback to "tight" if personality unknown
        p_templates = templates.get(personality, templates["tight"])
        options = p_templates.get(action, ["..."])

        return random.choice(options)

    def _call_llm(self, player: Player, action: ActionType, state: GameState) -> str:
        # Placeholder for actual API call
        # prompt = f"You are {player.name}, a {player.personality} poker player. You just did {action.name}. Say something short and witty."
        return "Thinking..."
