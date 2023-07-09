import os
import pickle
import time
import dataclasses
from typing import List

@dataclasses.dataclass
class HandData:
  hand: List[List[float]]
  world_hand: List[List[float]]
  handedness: List[float]

class Recorder:
    def __init__(self) -> None:
        self.recording = False
        self.label = None
        self.data = {}
        self.capacity = 100
    
    def start_recording(self):
        self.recording = True
        self.start_time = int(time.time())

    def stop_recording(self):
        self.recording = False
    
    def change_label(self, label):
        self.label = label
        self.stop_recording()

    def label_count(self):
        if not self.label in self.data:
            return 0
        return len(self.data[self.label])
    
    def record_gesture(self, data):
        if data.hand_landmarks and self.label != None:
            hand_landmarks = [[hand_landmark.x, hand_landmark.y, hand_landmark.z]
                            for hand_landmark in data.hand_landmarks[0]]
            hand_world_landmarks = [[
                hand_landmark.x, hand_landmark.y, hand_landmark.z
            ] for hand_landmark in data.hand_world_landmarks[0]]
            handedness_scores = [
                handedness.score for handedness in data.handedness[0]
            ]

            hand_data = HandData(
              hand=hand_landmarks,
              world_hand=hand_world_landmarks,
              handedness=handedness_scores)
            
            if self.label in self.data:
                self.data[self.label].append(hand_data)
            else:
                self.data[self.label] = [hand_data]

            if len(self.data[self.label]) % self.capacity == 0:
                self.flush()

    def flush(self):
        self.stop_recording()
        assert self.label != None
        directory = os.path.join('raw_data', self.label)
        if not os.path.exists(directory):
            os.makedirs(directory)
            print('Created directory: ', directory)
        
        file = os.path.join(directory, str(int(time.time())))

        pickle.dump(self.data[self.label], open(file, 'wb'))

        self.data.pop(self.label)   # reset data

    def toggle_recording(self):
        if self.recording or self.label == None:
            self.stop_recording()
        else:
            self.start_recording()