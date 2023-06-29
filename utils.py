import time
import mediapipe as mp
import cv2 as cv

class FPSLogger:
    def __init__(self) -> None:
        self.prev_frame_time = time.time()
        self.new_frame_time = time.time()

    def get_fps(self):
        self.new_frame_time = time.time()
        fps = 1 / (self.new_frame_time - self.prev_frame_time)
        self.prev_frame_time = self.new_frame_time
        return round(fps)
    
def draw_gesture(image, result):
    if result.hand_landmarks:
        # for drawing hand-bounding box
        min_x = image.shape[1]
        min_y = image.shape[0]
        max_x = 0 
        max_y = 0

        landmarks = result.hand_landmarks[0]
        for landmark in landmarks:
            # draw points on landmarks
            xy = denormalize_xy(landmark, image)
            cv.circle(image, xy, 5, (0, 0, 255), -1)

            # calculate hand-bounding box
            min_x = min(min_x, xy[0])
            min_y = min(min_y, xy[1])
            max_x = max(max_x, xy[0])
            max_y = max(max_y, xy[1])

        # draw hand-bounding box with label
        pt1 = (min_x, min_y)
        pt2 = (max_x, max_y)
        cv.rectangle(image, pt1, pt2, (225, 0, 0), 2) # type: ignore
        hand = result.handedness[0][0].category_name
        gesture = result.gestures[0][0].category_name
        text = hand + ": " + gesture
        cv.putText(image, text, (pt1[0], pt1[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 1, cv.LINE_AA)

        # draw lines between landmarks
        for connection in mp.tasks.vision.HandLandmarksConnections().HAND_CONNECTIONS:
            start = connection.start
            end = connection.end
            start_point = denormalize_xy(landmarks[start], image)
            end_point = denormalize_xy(landmarks[end], image)
            cv.line(image, start_point, end_point, (0, 255, 0), 3)

def denormalize_xy(landmark, image):
    image_width, image_height = image.shape[1], image.shape[0]
    x = int(landmark.x * image_width)
    y = int(landmark.y * image_height)
    return (x, y)