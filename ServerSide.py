from FaceDetection import *
from EdgeDevice import *
from Q_func import *
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def match_rewards(episode):
    print('Matching rewards')
    face_detect = FaceDetect()
    last_angle = -1
    reward = np.zeros(len(episode[0].state))
    for sample in episode:
        for i, frame in enumerate(sample.state):
            (boxes, scores, classes, num_detections) = face_detect.tDetector.run(frame)

            [h, w] = frame.shape[:2]

            centers, temp_reward = face_detect.calc_reward(boxes, h, w, scores)
            if temp_reward == 0:
                if last_angle == sample.angle:
                    temp_reward -= 1
            reward[i] = temp_reward

        sample.reward = np.max(reward)

        last_angle = sample.angle

        print('Reward =', reward)

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

        x_train.append(sample.state[0])
        y_train.append(y_sample)

    return x_train, y_train


if __name__ == '__main__':
    # trigger edge device to run and send episode
    print('First run, capture data')
    episode = capture(model=0, length=10, camID=1, run_number=0)

    # calculate reward for each state (depends on the state only)
    print('Match rewards')
    match_rewards(episode)

    # Calculate q function errors and generate learning data set
    print('Calc Errors')
    x, y = calc_errors(episode)

    # perform learning step on network
    print('Load updated model')
    model = load_model([x, y])

    print('Update Q function')
    model = Q_func_update([x, y], model, 20)

    # send new model to edge device
    for i in range(15):
        print('------------------------------------------------------------------------------------------')
        print('Iteration #', i)
        episode = capture(model, length=50, camID=1, run_number=i+1)

        # calculate reward for each state (depends on the state only)
        match_rewards(episode)

        # Calculate q function errors and generate learning data set
        x, y = calc_errors(episode)
        model = Q_func_update([x, y], model, 20)

    tf.keras.models.save_model(model, './trained_model', overwrite=True)

    print('Done')



