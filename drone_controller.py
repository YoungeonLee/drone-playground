from djitellopy import Tello
from pynput import keyboard
from enum import Enum

class Direction(Enum):
    STOP = 0
    LEFT = 1
    RIGHT = 2
    FORWARD = 3
    BACKWARD = 4
    UP = 5
    DOWN = 6
    CLOCK = 7
    COUNTER_CLOCK = 8

class DroneController:
    def __init__(self, drone: Tello, mode='keyboard') -> None:
        self.drone = drone
        self.mode = mode
        self.flying = False

        # listen to keyboard for controlling the drone
        def on_press(key):
            if self.flying:
                try:
                    self.keyboard_control(key.char)
                except AttributeError:
                    print('special key {0} pressed'.format(key))
        def on_release(key):
            if self.flying:
                self.keyboard_control(-1)
        listener = keyboard.Listener(
            on_press=on_press,
            on_release=on_release)  # type: ignore
        listener.start()
    
    def control_drone(self, key, result_holder):
        # toggle flying
        if key == 32:
            if self.flying:
                self.land()
            else:
                self.fly()

        # toggle gesutre mode
        if key == ord('g'):
            if self.mode == 'gesture':
                self.mode = 'keyboard'
            else:
                self.mode = 'gesture'

        # gesture control
        if self.mode == 'gesture':
            if result_holder.has_handlandmark_results():
                self.gesture_control(result_holder.gesture)
            else:
                self.keyboard_control(-1)

    def keyboard_control(self, key):
        speed = 50
        if key == 'a':
            self.drone.send_rc_control(-speed, 0, 0, 0)
        elif key == 'd':
            self.drone.send_rc_control(speed, 0, 0, 0)
        elif key == 'w':
            self.drone.send_rc_control(0, speed, 0, 0)
        elif key == 's':
            self.drone.send_rc_control(0, -speed, 0, 0)
        elif key == 'e':
            self.drone.send_rc_control(0, 0, 0, speed)
        elif key == 'q':
            self.drone.send_rc_control(0, 0, 0, -speed)
        elif key == 'r':
            self.drone.send_rc_control(0, 0, speed, 0)
        elif key == 'f':
            self.drone.send_rc_control(0, 0, -speed, 0)
        else:
            self.drone.send_rc_control(0, 0, 0, 0)

    def gesture_control(self, gesture):
        print(gesture)
        if gesture['prediction'] == 'left':
            self.keyboard_control('d')
        elif gesture['prediction'] == 'right':
            self.keyboard_control('a')
        elif gesture['prediction'] == 'forward':
            self.keyboard_control('s')
        elif gesture['prediction'] == 'backward':
            self.keyboard_control('w')
        elif gesture['prediction'] == 'up':
            self.keyboard_control('r')
        elif gesture['prediction'] == 'down':
            self.keyboard_control('f')
        else:
            self.keyboard_control(-1)
    
    def land(self):
        self.drone.land()
        self.flying = False

    def fly(self):
        self.drone.takeoff()
        self.flying = True