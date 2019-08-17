from FaceDetection import *
from EdgeDevice import *
from Q_func import *

def match_rewards(episode):
    print('Matching rewards')
    face_detect = FaceDetect()

    for sample in episode:
        (boxes, scores, classes, num_detections) = face_detect.tDetector.run(sample.state)

        [h, w] = sample.state.shape[:2]

        centers, reward = face_detect.calc_reward(boxes, h, w, scores)

        sample.reward = reward

        print('Reward =', reward)

    print('Completed reward calculation')


def calc_errors(episode):
    print('Calculating errors')
    x_train = []
    y_train = []

    for sample in episode[:-1]:
        Qtag = sample.reward - np.max(episode[sample.next_state].Q_value)
        error = (Qtag - np.max(sample.Q_value))**2
        x_train.append(sample.state)
        y_train.append(error)

    return x_train, y_train


if __name__ == '__main__':
    # trigger edge device to run and send episode
    episode = capture(length=100, camID=0)

    # calculate reward for each state (depends on the state only)
    match_rewards(episode)

    # Calculate q function errors and generate learning data set
    x, y = calc_errors(episode)

    # perform learning step on network
    model = load_model([x, y])

    model = Q_func_update([x, y], model, 20)

    print('Done')



