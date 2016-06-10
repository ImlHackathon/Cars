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
# from simulator import AbstractPolicy

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
        state = np.zeros((20, 30))
        state[int(agent[X])][self._loc_to_matrix_y(agent[Y])] = 1

        for o in obstacles:
            if o[X] < 60:
                state[int(X/30)][self._loc_to_matrix_y(Y)] = 1

        return state.reshape(1,-1)

    def _loc_to_matrix_y(self, y):
        for i, dy in enumerate(range(11, 71, 2)):
            if y <= (dy/10.0):
                return i