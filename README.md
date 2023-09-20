# Requirements

- python3
- tello drone
- webcam (optional)

# Run Example

```bash
python -m pip install -r requirements.txt   # download dependencies
# connect computer to tello drone
python main.py -c tello -d                  # contorl tello with tello camera
```

# Arguments

`-c {tello,pc}, --camera {tello,pc}` chooses which camera to use

`-d, --drone, --no-drone` control drone if drone argument exists

## Example

```bash
python main.py -c tello -d  # use tello camera to control tello
python main.py -c pc -d     # use pc camera to control tello
python main.py -c tello     # use tello camera to only view hand gesture
python main.py -c pc        # use pc camera to only view hand gesture
```

# Keyboard inputs

Press `g` to toggle `keyboard` and `gesture` modes

Press `space` to land/fly drone (`-d` or `--drone` only)

## Keyboard mode

Use keyboard to control the drone

- `w` forward
- `s` backward
- `a` left
- `d` right
- `e` rotate right
- `q` rotate left
- `r` ascend
- `f` descend

## Gesture mode

Use hand gesture to control the drone

![left gesture](https://github.com/YoungeonLee/drone-playground/blob/main/pictures/left.jpg?raw=true)
![right gesture](https://github.com/YoungeonLee/drone-playground/blob/main/pictures/right.jpg?raw=true)
![forward gesture](https://github.com/YoungeonLee/drone-playground/blob/main/pictures/forward.jpg?raw=true)
![backward gesture](https://github.com/YoungeonLee/drone-playground/blob/main/pictures/backward.jpg?raw=true)
![up gesture](https://github.com/YoungeonLee/drone-playground/blob/main/pictures/up.jpg?raw=true)
![down gesture](https://github.com/YoungeonLee/drone-playground/blob/main/pictures/down.jpg?raw=true)
![stop gesture](https://github.com/YoungeonLee/drone-playground/blob/main/pictures/stop.jpg?raw=true)

## Recording

Press a number to set which gesture data you want to collect

- `1` record gesture `left`
- `2` record gesture `right`
- `3` record gesture `forward`
- `4` record gesture `backward`
- `5` record gesture `up`
- `6` record gesture `down`
- `7` record gesture `stop`
- `8` record gesture `none`

Press `c` to record/capture 100 frames (only when camera is pc)

The data is raw result from mediapipe's handlandmarks and stored to `raw_data` folder

# Training

Running the notebook `train.ipynb` uses files in `raw_data` to train a simple neural network
and saves it to `gesture_recognizer.keras`
