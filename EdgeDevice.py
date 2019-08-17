import cv2
import time
import numpy as np


class Sample:
    def __init__(self, time, index, state, action, Qvalue):
        self.time = time
        self.index = index
        self.state = state
        self.action = action
        self.Q_value = Qvalue
        self.reward = 0
        self.next_state = self.index + 1

    def get_reward(self):
        self.reward = 0


def capture(length, camID=0):

    camera = cv2.VideoCapture(camID)
    epoch = []

    print('Capturing video')
    for pic in range(length):
        ret, image = camera.read()
        current_time = time.time()

        if ret == 0:
            print('ret = 0 for some reason')
            break

        [h, w] = image.shape[:2]

        image = cv2.flip(image, 1)

        Q_value = Q_function(image)

        action = epsilon_greedy(np.argmax(Q_value))

        epoch.append(Sample(current_time, pic, image, action, Q_value))

    return epoch


def Q_function(state, n_actions=3):
    # insert some sort of neural network to predict action from state

    action = np.random.rand(n_actions)
    action = action/np.linalg.norm(action)

    return action


def epsilon_greedy(action, epsilon=0.2):

    if np.random.rand() > epsilon:
        return action
    else:
        return np.random.randint(0, 2)





# if __name__ == "__main__":
#     episode = capture(100, 0)
#     print('Done')
