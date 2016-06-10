#!/usr/bin/python
"""
===================================================
     Introduction to Machine Learning (67577)
             IML HACKATHON, June 2016

            **  Driving Simulator  **

===================================================
"""
#===============================================================================
# Imports
#===============================================================================
import argparse
import sys
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

# import student_policy as sp


numbers = {}
#===============================================================================
# Consts
#===============================================================================
LANE_LENGTH = 100.0
LANE_WIDTH = 8.0
NUM_OBSTACLES = 4
CAR_LENGTH = 5.0
CAR_WIDTH = 2.0
DX_SIZE = 1.0
DY_SIZE = 0.2
#------------------------------------------------------------------------------
CLR_WHITE = (255,255,255)
CLR_GRAY = (100.0/255,100.0/255,100.0/255)
CLR_CAR = (1, 0, 0)
CLR_OBSTACLE = (0, 0, 1)

#===============================================================================
# Box, Obstacle, Agent
#===============================================================================

class Box():
    def __init__(self, simulator, color):
        self._simulator = simulator
        self._rect = None
        self._color = color

    def draw(self):
        if self._rect == None:
            self._rect = patches.Rectangle((self._X - CAR_LENGTH/2, self._Y - CAR_WIDTH/2), CAR_LENGTH, CAR_WIDTH, color=self._color)
            self._simulator._ax.add_patch(self._rect)
        else:
            self._rect.set_xy((self._X - CAR_LENGTH/2, self._Y - CAR_WIDTH/2))


class Obstacle(Box):
    def __init__(self, simulator,t):
        Box.__init__(self, simulator, CLR_OBSTACLE)
        self._X = int(((t+0.25+np.random.rand()*0.5)*LANE_LENGTH/NUM_OBSTACLES)/DX_SIZE)*DX_SIZE
        self._Y = CAR_WIDTH/2 + int((np.random.rand())*(LANE_WIDTH-CAR_WIDTH)/DY_SIZE)*DY_SIZE

    def move(self):
        self._X -= DX_SIZE
        if self._X < 0:
            self._X = LANE_LENGTH-1
            self._Y = CAR_WIDTH/2 + int((np.random.rand())*(LANE_WIDTH-CAR_WIDTH)/DY_SIZE)*DY_SIZE
            # if str(self._Y) not in numbers.keys():
            #     numbers[str(self._X)] = 1
            # else:
            #     numbers[str(self._X)] += 1
                # arr = [k for k in numbers.keys()]
                # arr.sort()
                # print(str(len(arr)) + " " + str(arr))
        # print (self._X)

class Agent(Box):
    def __init__(self, simulator, policy):
        Box.__init__(self, simulator,CLR_CAR)
        self._policy = policy
        self._game_over = False
        self._X = CAR_LENGTH
        self._Y = CAR_WIDTH/2 + int((np.random.rand())*(LANE_WIDTH-CAR_WIDTH)/DY_SIZE)*DY_SIZE


    def move(self):
        desired_move = DY_SIZE*np.sign(self._policy.get((self._X,self._Y),[(o._X,o._Y) for o in self._simulator._obstacles]))
        self._Y += desired_move
        self._Y = max(CAR_WIDTH/2,min(LANE_WIDTH-CAR_WIDTH/2,self._Y))
        return desired_move

    def move_man(self, action):
        desired_move = DY_SIZE*np.sign(action)
        self._Y += desired_move
        self._Y = max(CAR_WIDTH/2,min(LANE_WIDTH-CAR_WIDTH/2,self._Y))
        return desired_move

#===============================================================================
# Policy
#===============================================================================
class AbstractPolicy():
    def get(self,agent_pos,obstacles_pos):
        # agent_pos is tuple, (X,Y), determining location of the agent
        # obstacles_pos is list of tuples of the location of the obstacles
        # return value should be 0, -1, or 1 (determining if to stay go up or down on the Y axis)
        raise "Not Implemented"

class NaivePolicy(AbstractPolicy):
    def get(self,agent_pos,obstacles_pos):
        relevant_obstacles = [o for o in obstacles_pos if o[0] > agent_pos[0]]
        if not relevant_obstacles:
            return 0
        closest_obstacle = relevant_obstacles[np.argmin(np.array([o[0] for o in relevant_obstacles]))]
        y_dist = closest_obstacle[1] - agent_pos[1]
        if abs(y_dist) > 1.1*CAR_WIDTH:
            return 0
        else:
            if (y_dist > 0 and closest_obstacle[1] >= 1.5*CAR_WIDTH) or (closest_obstacle[1] >= LANE_WIDTH-1.5*CAR_WIDTH):
                return -1
            else:
                return 1

#===============================================================================
# Simulator
#===============================================================================
class Simulator(object):
    def __init__(self, args):
        self._args = args
        self._screen = None
        self._game_round = 0
        self._total_reward = 0
        self._build_policy()
        self._build_state()
        self._init_gui()


    def _init_gui(self):
        self._screen, self._ax = plt.subplots(figsize=(16, 18*LANE_WIDTH/LANE_LENGTH))
        self._screen.canvas.set_window_title('Car Simulator ' + self._args.policy)
        self._ax.axis([0, LANE_LENGTH, 0, LANE_WIDTH])
        plt.ion()
        self._ax.set_aspect(1)
        self._ax.set_axis_bgcolor(CLR_GRAY)
        self._running = True

    def reset(self, args):
        if self._screen is not None:
            plt.close('all')
        self._args = args
        self._screen = None
        self._game_round = 0
        self._total_reward = 0
        self._build_policy()
        self._build_state()
        self._init_gui()

    def run(self):
        sys.stderr.write('Game round: 0')

        while self._game_round < self._args.rounds:
            self._game_round += 1
            action = self._agent.move()
            o_in_cond = "place holder"
            for o in self._obstacles:
                o.move()
            accident = [((np.abs(o._X-self._agent._X)<CAR_LENGTH) and (np.abs(o._Y-self._agent._Y)<CAR_WIDTH)) for o in self._obstacles]
            self._current_reward = 1.0*((not any(accident))) - 0.01*(action != 0)
            self._total_reward += self._current_reward
            if (self._game_round % self._args.interval_gui == 0):
                self._on_render()
                pass

    def _build_policy(self):
        if self._args.policy == "NaivePolicy":
            self._policy = NaivePolicy()
        elif self._args.policy == "StudentPolicy":
            self._policy = sp.StudentPolicy()
        else:
            assert "Unknown policy"

    def _build_state(self):
        self._agent = Agent(self,self._policy)
        self._obstacles = [Obstacle(self,t) for t in range(NUM_OBSTACLES)]


    def _on_render(self):
        self._agent.draw()
        for o in self._obstacles: o.draw()
        plt.title('Round: %d, Current Reward %.2f, Avg Reward: %.2f' %  ((self._game_round), (self._current_reward), (self._total_reward/(self._game_round+1))))
        plt.pause(1.0/self._args.fps)

    def act(self, action):
        try:
            action = action[0]
        except:
            action = action
        action = action - 1
        self._game_round += 1
        self._agent.move_man(action)
        in_range = None
        for o in self._obstacles:
            if o._X <= 20 and o._X >= 10:
                in_range = o
            o.move()
        # accident = [(np.abs(o._X-self._agent._X)<CAR_LENGTH) and (np.abs(o._Y-self._agent._Y)<CAR_WIDTH) for o in self._obstacles]
        o_accident = [(o, ((np.abs(o._X-self._agent._X)<CAR_LENGTH) and (np.abs(o._Y-self._agent._Y)<CAR_WIDTH))) for o in self._obstacles]
        # self._current_reward = 1.0*((not any(accident))) - 0.01*(action != 0)
        if action == 0:
            self._current_reward = 0.01
        else:
            self._current_reward = 0
        # self._total_reward += self._current_reward

        print("\t\treward: " + str(self._current_reward))

        if (self._game_round % self._args.interval_gui == 0):
            self._on_render()

        return self.observe(), self._current_reward, self._game_round >= self._args.rounds

    def observe(self):
        ROWS = 5
        COLS = 5
        np.set_printoptions(threshold=np.nan)
        state = np.zeros((COLS, ROWS))

        for y, j in enumerate(range(1, ROWS+1)):
            if self._agent._Y < ((j * LANE_WIDTH)/ROWS):
                state[0][y] = 1
                break


        for o in self._obstacles:
            for x, i in enumerate(range(1, COLS)):
                if o._X < ((i * LANE_LENGTH)/(COLS)):
                    for y, j in enumerate(range(1, ROWS+1)):
                        if o._Y < ((j * LANE_WIDTH)/ROWS):
                            state[x+1][y] = 1
                            break

        print(state.transpose())

        return state.reshape(1,-1)


#===============================================================================
# Main
#===============================================================================
def get_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p","--policy", default='StudentPolicy', help='Specify a policy name')
    parser.add_argument("-f", "--fps", type=float, default=10, help='Set time sleep')
    parser.add_argument("-i", "--interval_gui", type=int, default=1, help='Interval between gui presentations')
    parser.add_argument("-r", "--rounds", type=int, default=10000, help='Number of games rounds')

    args = parser.parse_args()
    return args


def main(args):
    # Run simulator
    simulator = Simulator(args)
    simulator.run()


#------------------------------------------------------------------------------
if __name__ == '__main__':
    args = get_command_line_args()
    main(args)




