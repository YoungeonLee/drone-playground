import mediapipe as mp
import cv2 as cv
import time
from utils import FPSLogger, draw_landmarks
from recorder import Recorder
from gesture_predictor import GesturePredictor
from result_holder import ResultHolder
import argparse
from djitellopy import Tello
import numpy as np
from drone_controller import DroneController

parser = argparse.ArgumentParser(
                    prog='main.py',
                    description='Control drone with various methods')
parser.add_argument('-c', '--camera', default="tello", choices=["tello", "pc"], help="which camera to use") # positional argument
parser.add_argument('-d', '--drone', action=argparse.BooleanOptionalAction, help="control drone if true")

args = parser.parse_args()

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode
# PoseLandmarker = mp.tasks.vision.PoseLandmarker
# PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
# PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult

recorder = Recorder()
capture = None
tello = None

# start video capture
drone_controller = None
if args.camera == "tello":
    tello = Tello()
    tello.connect()
    tello.streamon()
    frame_read = tello.get_frame_read()
elif args.camera == "pc":
    capture = cv.VideoCapture(0)
    if not capture.isOpened():
        print("Cannot open camera")
        exit()
else:
    print('Error: Unknown camera argument')
    exit()

if args.drone:
    if not tello:
        tello = Tello()
        tello.connect()
    drone_controller = DroneController(tello)


result_holder = ResultHolder()

fps_logger = FPSLogger()

gesture_predictor = GesturePredictor()

def update_handlandmarker_result(res: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    """callback function that runs after processing an image"""
    gesture = gesture_predictor.predict(res)    # type: ignore
    result_holder.update_handlandmark(res, gesture)

# def update_poselandmark_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
#     result_holder.update_poselandmark(result)
#     print('pose landmarker result: {}'.format(result))

handlandmarker_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=update_handlandmarker_result)
# poselandmark_options = PoseLandmarkerOptions(
#     base_options=BaseOptions(model_asset_path='pose_landmarker_heavy.task'),
#     running_mode=VisionRunningMode.LIVE_STREAM,
#     result_callback=update_poselandmark_result)

try:
    with HandLandmarker.create_from_options(handlandmarker_options) as hand_landmarker:
        # with PoseLandmarker.create_from_options(poselandmark_options) as pose_landmarker:
        while True:
            # read a frame from camera
            if args.camera == "tello":
                assert frame_read   # type: ignore
                frame = frame_read.frame
                assert type(frame) == np.ndarray
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            elif args.camera == "pc":
                ret, frame = capture.read() # type: ignore
                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break
            image = cv.flip(frame, 1)   # flip image horizontally   # type: ignore 
            time_ms = int(round(time.time() * 1000))
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            hand_landmarker.detect_async(mp_image, time_ms)
            # pose_landmarker.detect_async(mp_image, time_ms)

            # Display the resulting frame
            # write FPS and battery
            text = "FPS: " + str(fps_logger.get_fps())
            if args.drone:
                text += f' Battery: {tello.get_battery()}' # type: ignore
            cv.putText(image, text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 1, cv.LINE_AA)
            if recorder.label != None:
                text = f"Gesture: {recorder.label} {recorder.label_count()}" 
                cv.putText(image, text, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 1, cv.LINE_AA)
                if recorder.recording and result_holder.has_handlandmark_results():
                    recorder.record_gesture(result_holder.handlandmark)   # record if needed

            draw_landmarks(image, result_holder) # draw detected hand gesture
                
            cv.imshow('frame', image)   # show image

            key = cv.waitKey(1)
            num = key - 48
            if args.drone:
                assert drone_controller # type: ignore
                drone_controller.control_drone(key, result_holder)
            # press esc to exit
            if key == 27:
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
            # press c toggle record
            elif key == ord('c'):
                if args.camera == 'pc':
                    recorder.toggle_recording()
finally:
    if drone_controller:
        drone_controller.land()

# When everything done, release the capture
if capture: 
    capture.release()
cv.destroyAllWindows()