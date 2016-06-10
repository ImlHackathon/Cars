"""
===================================================
     Introduction to Machine Learning (67577)
             IML HACKATHON, June 2016

            **  Autonomous Driver  **

Auther(s):

===================================================
"""


import json
import numpy as np
from keras.models import model_from_json
from simulator import *

class StudentPolicy(AbstractPolicy):

    def __init__(self):
        with open("model.json", "r") as jfile:
            self._model = model_from_json(json.load(jfile))
        self._model.load_weights("model.h5")
        self._model.compile("sgd", "mse")

    def get(self,agent_pos,obstacles_pos):
        q = self._model.predict(self._observe(agent_pos, obstacles_pos))
        action = np.argmax(q[0])
        print(action-1)
        return action - 1

    def _observe(self, agent, obstacles):

        X, Y = (0, 1)
        ROWS = 5
        COLS = 5
        state = np.zeros((COLS, ROWS))

        for y, j in enumerate(range(1, ROWS+1)):
            if agent[Y] < ((j * LANE_WIDTH)/ROWS):
                state[0][y] = 1
                break


        for o in obstacles:
            for x, i in enumerate(range(1, COLS)):
                if o[X] < ((i * LANE_LENGTH)/(COLS)):
                    for y, j in enumerate(range(1, ROWS+1)):
                        if o[Y] < ((j * LANE_WIDTH)/ROWS):
                            state[x+1][y] = 1
                            break

        return state.reshape(1,-1)

    def _loc_to_matrix_y(self, y):
        for i, dy in enumerate(range(11, 71, 2)):
            if y <= (dy/10.0):
                return i