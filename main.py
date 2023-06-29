import mediapipe as mp
import cv2 as cv
import threading
import time
from utils import FPSLogger, draw_gesture

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# start video capture
capture = cv.VideoCapture(0)
if not capture.isOpened():
    print("Cannot open camera")
    exit()

# thread to wait for image recognition
working = threading.Event()
global result

fps_logger = FPSLogger()

def update_result(res: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    """callback function that runs after processing an image"""
    global result
    result = res
    working.set()

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='gesture_recognizer.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=update_result)
with GestureRecognizer.create_from_options(options) as recognizer:
    while True:
        # read a frame from camera
        ret, frame = capture.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        image = cv.flip(frame, 1)   # flip image horizontally
        time_ms = int(round(time.time() * 1000))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        recognizer.recognize_async(mp_image, time_ms)
        working.wait()  # wait until the image is processed

        # Display the resulting frame
        # write FPS
        text = "FPS: " + str(fps_logger.get_fps())
        cv.putText(image, text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 1, cv.LINE_AA)
        draw_gesture(image, result) # draw detected hand gesture
        cv.imshow('frame', image)   # show image

        # press q to exit
        if cv.waitKey(1) == ord('q'):
            break

# When everything done, release the capture
capture.release()
cv.destroyAllWindows()