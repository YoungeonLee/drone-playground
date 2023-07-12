from typing import Any
class ResultHolder:
    def __init__(self) -> None:
        self.handlandmark: Any = None
        self.gesture: Any = None
        self.poselandmark: Any = None

    def update_handlandmark(self, handlandmark, gesture):
        self.handlandmark = handlandmark
        self.gesture = gesture

    def update_poselandmark(self, res):
        self.poselandmark = res
    
    def has_handlandmark_results(self):
        return self.handlandmark != None and self.gesture != None
    
    def has_poselandmark_result(self):
        return self.poselandmark != None