from typing import Any
class ResultHolder:
    def __init__(self) -> None:
        self.handlandmark: Any = None
        self.gesture: Any = None

    def update(self, handlandmark, gesture):
        self.handlandmark = handlandmark
        self.gesture = gesture
    
    def has_results(self):
        return self.handlandmark != None and self.gesture != None