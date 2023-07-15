import os
import time
import mediapipe as mp
import cv2 as cv
import numpy as np
from result_holder import ResultHolder

THREE_POINTS = [0, 5, 17]

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
        landmarks = hand_result.hand_landmarks[0]

        # for drawing hand-bounding box
        min_x = image.shape[1]
        min_y = image.shape[0]
        max_x = 0 
        max_y = 0

        # for center point
        mean_x = 0
        mean_y = 0
        angle = compute_angles(landmarks)
        for i in THREE_POINTS:
            mean_x += landmarks[i].x
            mean_y += landmarks[i].y
        mean_x /= len(THREE_POINTS)
        mean_y /= len(THREE_POINTS)

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
        text += f' {angle}'
        cv.putText(image, text, (pt1[0], pt1[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 1, color, 1, cv.LINE_AA)

        # draw lines between landmarks
        for connection in mp.tasks.vision.HandLandmarksConnections().HAND_CONNECTIONS:
            start = connection.start
            end = connection.end
            start_point = denormalize_xy(landmarks[start], image)
            end_point = denormalize_xy(landmarks[end], image)
            cv.line(image, start_point, end_point, (0, 255, 0), 3)

        # calculate mean
        x, y = denormalize_xy_raw(mean_x, mean_y, image)
        cv.circle(image, (x, y), 2, (255, 0, 0))

def calculate_normal_vector(point1, point2, point3):
    vector1 = point2 - point1
    vector2 = point3 - point1
    normal_vector = np.cross(vector1, vector2)
    return normal_vector

def angle_from_two_vectors(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    angle_rad = np.arccos(dot_product / (magnitude1 * magnitude2))
    angle_deg = np.degrees(angle_rad)
    return round(angle_deg)

def compute_angles(hand_landmarks):
    point1 = hand_landmarks[THREE_POINTS[0]]
    point1 = np.array((point1.x, point1.y, point1.z))
    point2 = hand_landmarks[THREE_POINTS[1]]
    point2 = np.array((point2.x, point2.y, point2.z))
    point3 = hand_landmarks[THREE_POINTS[2]]
    point3 = np.array((point3.x, point3.y, point3.z))
    print(point1, point2, point3)
    normal = calculate_normal_vector(point1, point2, point3)

    x_axis = np.array((1, 0, 0))
    y_axis = np.array((0, 1, 0))

    x_angle = angle_from_two_vectors(x_axis, normal)
    y_angle = angle_from_two_vectors(y_axis, normal)

    return x_angle, y_angle

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

def denormalize_xy_raw(x, y, image):
    image_width, image_height = image.shape[1], image.shape[0]
    x = int(x * image_width)
    y = int(y * image_height)
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