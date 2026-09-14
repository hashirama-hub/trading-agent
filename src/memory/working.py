from typing import List, Dict
from datetime import datetime


class WorkingMemory:
    def __init__(self, max_context_turns: int = 20):
        self.context_turns: List[Dict] = []
        self.max_context_turns = max_context_turns

    def add_turn(self, thought: str, action: str, result: str):
        self.context_turns.append({
            "timestamp": datetime.utcnow().isoformat(),
            "thought": thought,
            "action": action,
            "result": result,
        })
        if len(self.context_turns) > self.max_context_turns:
            self.context_turns.pop(0)

    def get_context(self) -> str:
        return "\n".join([
            f"Turn {i}: thought={t['thought']}, action={t['action']}, result={t['result']}"
            for i, t in enumerate(self.context_turns[-10:])
        ])

    def summarize(self) -> str:
        if len(self.context_turns) <= self.max_context_turns:
            return self.get_context()
        recent = self.context_turns[-5:]
        summary = f"Earlier {len(self.context_turns) - 5} turns summarized"
        return summary + "\n" + self.get_context()

    def clear(self):
        self.context_turns.clear()

    def get_turn_count(self) -> int:
        return len(self.context_turns)