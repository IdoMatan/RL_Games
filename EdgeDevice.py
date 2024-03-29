import cv2
import time
import numpy as np
import ArduinoServoControl
import tensorflow as tf


class Sample:
    def __init__(self, time, index, state, action, Qvalue, current_angle):
        self.time = time
        self.index = index
        self.state = state
        self.action = action
        self.Q_value = Qvalue
        self.reward = 0
        self.next_state = self.index + 1
        self.angle = current_angle

    def get_reward(self):
        self.reward = 0


class Q_function:
    def __init__(self, model):
        self.model = model
        print('Q function initialized')

    def predict_action(self, state):
        IMG_SIZE = 160
        # state = tf.cast(state, tf.float32)
        # state = (state / 127.5) - 1
        # state = tf.image.resize(state, (IMG_SIZE, IMG_SIZE))

        # state = tf.cast(state, tf.float32)
        state = (state / 127.5) - 1
        state = cv2.resize(state, (IMG_SIZE, IMG_SIZE))

        action = self.model.predict(state[np.newaxis])    # action is a 3x1 vector (cell for each possible action)
        return action

    def update_model(self, new_model):
        self.model = new_model


def capture(model, length, camID=1, run_number=0):
    print('Initializing camera')
    camera = cv2.VideoCapture(camID)
    print('Initializing servo')
    try:
        servo = ArduinoServoControl.ServoControl(
            port='/dev/ttyACM0',
            baudrate=9600,
            start_angle=np.random.randint(0, 180))

    except:
        print('trying Mac port')
        servo = ArduinoServoControl.ServoControl(
            port='/dev/cu.usbmodem14201',
            baudrate=9600,
            start_angle=np.random.randint(0, 180))
    else:
        print('Using Ubuntu port')

    if run_number > 0:
        Q_func = Q_function(model)

    epoch = []

    print('Capturing video')
    for pic in range(length):
        resize_shape = (700, 700)
        n_images = 3
        image_array = [np.zeros(resize_shape)]*n_images

        for frame in range(n_images):
            ret, image = camera.read()

            if ret == 0:
                print('ret = 0 for some reason')
                break

            current_time = time.time()

            image = cv2.resize(image, resize_shape)
            [h, w] = image.shape[:2]
            image = cv2.flip(image, 1)

            image_array[frame] = image

        if run_number > 0:
            Q_value = Q_func.predict_action(image)              # return 3 possible options (actions and their score)
            action = epsilon_greedy(np.argmax(Q_value),
                                    epsilon=np.exp(-run_number*0.05))   # Choose argmax action (or epsilon greedy)
            # action = weighted_selection(Q_value)

        else:
            action = pure_exploration()
            Q_value = np.zeros((1, 3))

        servo.execute_action(action, dtheta=5)
        time.sleep(0.75)

        # epoch.append(Sample(current_time, pic, image, action, Q_value, servo.current_angle))
        epoch.append(Sample(current_time, pic, image_array, action, Q_value, servo.current_angle))

    return epoch


def pure_exploration():
    # action = np.random.randint(0, 2)
    action = 0
    return action


def epsilon_greedy(action, epsilon=0.5):

    if np.random.rand() > epsilon:
        return action
    else:
        return np.random.randint(0, 2)


def weighted_selection(Q_values):
    probs = np.abs(Q_values[0] / np.sum(np.abs(Q_values[0])))

    action = np.random.choice(3, 1, p=probs)

    return int(action)


def deploy(model, camID=0):
    print('Initializing camera')
    camera = cv2.VideoCapture(camID)
    print('Initializing servo')

    try:
        servo = ArduinoServoControl.ServoControl(
            port='/dev/ttyACM0',
            baudrate=9600,
            start_angle=np.random.randint(0, 180))

    except:
        print('trying Mac port')
        servo = ArduinoServoControl.ServoControl(
            port='/dev/cu.usbmodem14201',
            baudrate=9600,
            start_angle=np.random.randint(0, 180))
    else:
        print('Using Ubuntu port')

    Q_func = Q_function(model)

    print('Capturing video')
    while True:
        ret, image = camera.read()
        current_time = time.time()

        if ret == 0:
            print('ret = 0 for some reason')
            break

        image = cv2.resize(image, (500, 500))
        [h, w] = image.shape[:2]

        image = cv2.flip(image, 1)

        Q_value = Q_func.predict_action(image)              # return 3 possible options (actions and their score)

        print(Q_value)
        action = np.argmax(Q_value)        # Choose argmax action (or epsilon greedy)


        servo.execute_action(action, dtheta=5)
        # time.sleep(0.5)


# if __name__ == "__main__":
#     episode = capture(100, 0)
#     print('Done')
