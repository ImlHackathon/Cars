import argparse
import json
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd
from simulator import Simulator
# import student_policy as sp

class ExperienceReplay(object):
    def __init__(self, max_memory=100, discount=.9):
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount

    def remember(self, states, game_over):
        # memory[i] = [[state_t, action_t, reward_t, state_t+1], game_over?]
        self.memory.append([states, game_over])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, model, batch_size=10):
        len_memory = len(self.memory)
        num_actions = model.output_shape[-1]
        env_dim = self.memory[0][0][0].shape[1]
        inputs = np.zeros((min(len_memory, batch_size), env_dim))
        targets = np.zeros((inputs.shape[0], num_actions))
        for i, idx in enumerate(np.random.randint(0, len_memory,
                                                  size=inputs.shape[0])):
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]
            game_over = self.memory[idx][1]

            inputs[i:i+1] = state_t
            # There should be no target values for actions not taken.
            # Thou shalt not correct actions not taken #deep
            targets[i] = model.predict(state_t)[0]
            Q_sa = np.max(model.predict(state_tp1)[0])
            if game_over:  # if game_over is True
                targets[i, action_t] = reward_t
            else:
                # reward_t + gamma * max_a' Q(s', a')
                targets[i, action_t] = reward_t + self.discount * Q_sa
        return inputs, targets

def get_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p","--policy", default='StudentPolicy', help='Specify a policy name')
    parser.add_argument("-f", "--fps", type=float, default=10, help='Set time sleep')
    parser.add_argument("-i", "--interval_gui", type=int, default=1, help='Interval between gui presentations')
    parser.add_argument("-r", "--rounds", type=int, default=10000, help='Number of games rounds')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    print("hey")
    # parameters
    epsilon = .3  # exploration
    num_actions = 3  # [move_up, stay, move_down]
    epoch = 500
    max_memory = 500
    hidden_size = 25
    batch_size = 60

    model = Sequential()
    model.add(Dense(15, input_shape=(15,), activation='tanh'))
    model.add(Dense(hidden_size, activation='tanh'))
    model.add(Dense(num_actions))
    model.compile(sgd(lr=.2), "mse")


    # If you want to continue training from a previous model, just uncomment the line bellow
    # model.load_weights("model.h5")

    # Define environment/game
    # env = Catch(grid_size)

    args = get_command_line_args()
    env = Simulator(args)

    # Initialize experience replay object
    exp_replay = ExperienceReplay(max_memory=max_memory)

    # Train
    win_cnt = 0
    for e in range(epoch):
        loss = 0.
        env.reset(args)
        # env.reset()
        game_over = False
        # get initial input
        input_t = env.observe()

        while not game_over:
            input_tm1 = input_t
            # get next action
            if np.random.rand() <= epsilon:
                action = np.random.randint(0, num_actions, size=1)
            else:
                q = model.predict(input_tm1)
                action = np.argmax(q[0])

            # apply action, get rewards and new state
            input_t, reward, game_over = env.act(action)
            if reward == 0:
                win_cnt += 1

            # store experience
            exp_replay.remember([input_tm1, action, reward, input_t], game_over)

            # adapt model
            inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)

            loss += model.train_on_batch(inputs, targets)
            # loss += model.train_on_batch(inputs, targets)[0]
        print("Epoch {:03d}/{:03d} | Loss {:.4f} | Win count {}".format(e, epoch, loss, win_cnt))

        if (e % 10 == 0):
            # Save trained model weights and architecture, this will be used by the visualization code
            print("saving model")
            model.save_weights("model.h5", overwrite=True)
            with open("model.json", "w") as outfile:
                json.dump(model.to_json(), outfile)

    print("saving model")
    model.save_weights("model.h5", overwrite=True)
    with open("model.json", "w") as outfile:
        json.dump(model.to_json(), outfile)


