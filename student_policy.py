
"""
===================================================
     Introduction to Machine Learning (67577)
             IML HACKATHON, June 2016
            **  Autonomous Driver  **
Auther(s): etzion, asaf; nachmana, gal; levanon, erez;
===================================================
"""

'''relevant imports'''
import json
from keras.models import model_from_json
from simulator import *


class StudentPolicy(AbstractPolicy):
    """
    a class that drives the car in the simulator according to
    a q learner we trained
    """

    def __init__(self):
        """
        the constructor function for the student policy object.
        loads the trained q learner from file
        """
        with open("model.json", "r") as jfile:
            self._model = model_from_json(json.load(jfile))
        self._model.load_weights("model.h5")
        self._model.compile("sgd", "mse")

    def get(self,agent_pos,obstacles_pos):
        """
        return the desired action according to a given position
        """
        q = self._model.predict(self._observe(agent_pos, obstacles_pos))
        action = np.argmax(q[0])
        action = action-1
        if action==1:
            action = -1
        elif action==-1:
            action = 1
        else:
            if agent_pos[1]<LANE_WIDTH/2:
                action = 1
            else:
                action = -1

        return action

    def _observe(self, agent, obstacles):
        """
        convert the situation given by simulator to the format expected by
        the learner
        """
        X, Y = (0, 1)
        ROWS = 3
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