from typing import Any
class ResultHolder:
    def __init__(self) -> None:
        self.handlandmark: Any = None
        self.gesture: Any = None

    def update(self, handlandmark, gesture):
        self.handlandmark = handlandmark
        self.gesture = gesture