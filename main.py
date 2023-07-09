import mediapipe as mp
import cv2 as cv
import threading
import time
from utils import FPSLogger, draw_gesture
import os
from recorder import Recorder
from gesture_predictor import GesturePredictor
from result_holder import ResultHolder

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

recorder = Recorder()

# start video capture
capture = cv.VideoCapture(0)
if not capture.isOpened():
    print("Cannot open camera")
    exit()

result_holder = ResultHolder()

fps_logger = FPSLogger()

gesture_predictor = GesturePredictor()

def update_result(res: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    """callback function that runs after processing an image"""
    gesture = gesture_predictor.predict(res)    # type: ignore
    result_holder.update(res, gesture)

model_path='hand_landmarker.task'
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=update_result)

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        # read a frame from camera
        ret, frame = capture.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        image = cv.flip(frame, 1)   # flip image horizontally
        time_ms = int(round(time.time() * 1000))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        landmarker.detect_async(mp_image, time_ms)

        # Display the resulting frame
        # write FPS
        text = "FPS: " + str(fps_logger.get_fps())
        cv.putText(image, text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 1, cv.LINE_AA)
        if recorder.label != None:
            text = f"Gesture: {recorder.label} {recorder.label_count()}" 
            cv.putText(image, text, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 1, cv.LINE_AA)
            if recorder.recording:
                recorder.record_gesture(result_holder.handlandmark)   # record if needed

        draw_gesture(image, result_holder) # draw detected hand gesture
            
        cv.imshow('frame', image)   # show image

        key = cv.waitKey(1)
        num = key - 48
        # press q to exit
        if key == ord('q'):
            break
        elif num == 1:
            recorder.change_label('left')
        elif num == 2:
            recorder.change_label('right')
        elif num == 3:
            recorder.change_label('forward')
        elif num == 4:
            recorder.change_label('backward')
        elif num == 5:
            recorder.change_label('up')
        elif num == 6:
            recorder.change_label('down')
        elif num == 7:
            recorder.change_label('stop')
        elif num == 8:
            recorder.change_label('none')
        # press space bar toggle record
        elif key == 32:
            recorder.toggle_recording()

# When everything done, release the capture
capture.release()
cv.destroyAllWindows()