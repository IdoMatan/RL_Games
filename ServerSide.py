from FaceDetection import *
from EdgeDevice import *
from Q_func import *
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


def match_rewards(episode):
    print('Matching rewards')
    face_detect = FaceDetect()

    for sample in episode:
        (boxes, scores, classes, num_detections) = face_detect.tDetector.run(sample.state)

        [h, w] = sample.state.shape[:2]

        centers, reward = face_detect.calc_reward(boxes, h, w, scores)

        sample.reward = reward - 1

        print('Reward =', reward-1)

    print('Completed reward calculation')


def calc_errors(episode):
    print('Calculating errors')
    x_train = []
    y_train = []

    for sample in episode[:-1]:
        Qtag = sample.reward + np.max(episode[sample.next_state].Q_value)   # new Q value w.r.t to sample

        y_sample = sample.Q_value             # use previous Q value and change only new executed action (next line)
        y_sample[:, int(sample.action)] = Qtag

        # error = (Qtag - np.max(sample.Q_value))**2
        # error = np.sign(Qtag - np.max(sample.Q_value)) + 1

        x_train.append(sample.state)
        y_train.append(y_sample)

    return x_train, y_train


if __name__ == '__main__':
    # trigger edge device to run and send episode
    print('First run, capture data')
    episode = capture(model=0, length=10, camID=0, first_run=1)

    # calculate reward for each state (depends on the state only)
    print('Match rewards')
    match_rewards(episode)

    # Calculate q function errors and generate learning data set
    print('Calc Errors')
    x, y = calc_errors(episode)

    # perform learning step on network
    print('load model')
    model = load_model([x, y])

    print('update Q function')
    model = Q_func_update([x, y], model, 20)

    # send new model to edge device
    for i in range(15):
        print('Iteration #', i)
        episode = capture(model, length=40, camID=0)

        # calculate reward for each state (depends on the state only)
        match_rewards(episode)

        # Calculate q function errors and generate learning data set
        x, y = calc_errors(episode)
        model = Q_func_update([x, y], model, 20)

    tf.keras.models.save_model(model, './trained_model', overwrite=True)

    print('Done')



