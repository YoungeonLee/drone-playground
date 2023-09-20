import numpy as np
import time
import mediapipe as mp
import cv2 as cv
from result_holder import ResultHolder

LABELS = {   
    'left': 0,
    'right': 1,
    'forward': 2,
    'backward': 3,
    'up': 4,
    'down': 5,
    'stop': 6,
    'none': 7
}

IDX_TO_LABLES = {
    0: 'left',
    1: 'right',
    2: 'forward',
    3: 'backward',
    4: 'up',
    5: 'down',
    6: 'stop',
    7: 'none'
}

class FPSLogger:
    def __init__(self) -> None:
        self.prev_frame_time = time.time()
        self.new_frame_time = time.time()

    def get_fps(self):
        self.new_frame_time = time.time()
        fps = 1 / (self.new_frame_time - self.prev_frame_time)
        self.prev_frame_time = self.new_frame_time
        return round(fps)
    
def draw_landmarks(image, result: ResultHolder):
    if result.has_handlandmark_results():
        draw_handlandmarks(image, result)
    
    if result.has_poselandmark_result():
        draw_poselandmarks(image, result)

def draw_handlandmarks(image, result: ResultHolder):
    hand_result = result.handlandmark
    if hand_result.hand_landmarks:
        # for drawing hand-bounding box
        min_x = image.shape[1]
        min_y = image.shape[0]
        max_x = 0 
        max_y = 0

        landmarks = hand_result.hand_landmarks[0]
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
        hand = hand_result.handedness[0][0].category_name
        gesture = result.gesture
        category = gesture['prediction']
        probability = round(gesture['probability'], 2)
        text = f"{hand}: {category} {str(probability)}"
        color = (100, 255, 0)
        if probability < 0.7:
            color = (0, 0, 255)
        cv.putText(image, text, (pt1[0], pt1[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 1, color, 1, cv.LINE_AA)

        # draw lines between landmarks
        for connection in mp.tasks.vision.HandLandmarksConnections().HAND_CONNECTIONS:
            start = connection.start
            end = connection.end
            start_point = denormalize_xy(landmarks[start], image)
            end_point = denormalize_xy(landmarks[end], image)
            cv.line(image, start_point, end_point, (0, 255, 0), 3)

def draw_poselandmarks(image, result: ResultHolder):
    pose_result = result.poselandmark
    if pose_result.pose_landmarks:
        # # for drawing hand-bounding box
        # min_x = image.shape[1]
        # min_y = image.shape[0]
        # max_x = 0 
        # max_y = 0

        landmarks = pose_result.pose_landmarks[0]
        for landmark in landmarks:
            # draw points on landmarks
            xy = denormalize_xy(landmark, image)
            cv.circle(image, xy, 5, (0, 0, 255), -1)

            # # calculate hand-bounding box
            # min_x = min(min_x, xy[0])
            # min_y = min(min_y, xy[1])
            # max_x = max(max_x, xy[0])
            # max_y = max(max_y, xy[1])

        # # draw hand-bounding box with label
        # pt1 = (min_x, min_y)
        # pt2 = (max_x, max_y)
        # cv.rectangle(image, pt1, pt2, (225, 0, 0), 2) # type: ignore
        # hand = pose_result.handedness[0][0].category_name
        # gesture = result.gesture
        # category = gesture['prediction']
        # probability = round(gesture['probability'], 2)
        # text = f"{hand}: {category} {str(probability)}"
        # color = (100, 255, 0)
        # if probability < 0.7:
            # color = (0, 0, 255)
        # cv.putText(image, text, (pt1[0], pt1[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 1, color, 1, cv.LINE_AA)

        # draw lines between landmarks
        for connection in mp.tasks.vision.PoseLandmarksConnections().POSE_LANDMARKS:
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

def preprocess_data(landmarks):
    processed_landmarks = []
    # set origin to 0
    # expand it
    min_x = float('inf')
    min_y = float('inf')
    min_z = float('inf')
    max_x = 0
    max_y = 0
    max_z = 0
    for point in landmarks:
        x = point[0]
        y = point[1]
        z = point[2]
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        min_z = min(min_z, z)
        max_x = max(max_x, x)
        max_y = max(max_y, y)
        max_z = max(max_z, z)

    offset_x = -min_x
    offset_y = -min_y
    offset_z = -min_z
    scaler_x = 1 / (max_x + offset_x)
    scaler_y = 1 / (max_y + offset_y)
    scaler_z = 1 / (max_z + offset_z)
    scaler = min(scaler_x, scaler_y)

    for point in landmarks:
        x = (point[0] + offset_x) * scaler
        y = (point[1] + offset_y) * scaler
        z = (point[2] + offset_z) * scaler_z
        processed_landmarks.append([x, y, z])

    return processed_landmarks  

# https://stackoverflow.com/a/56909036
def change_contrast(image, clip_hist_percent=1):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # Calculate grayscale histogram
    hist = cv.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)
    
    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))
    
    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0
    
    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1
    
    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1
    
    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    auto_result = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
    return auto_result