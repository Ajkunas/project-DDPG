import numpy as np

# Implementation of a heuristic policy for the pendulum

class HeuristicPendulumAgent:
    def __init__(self, env, torque):
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]
        self.env = env
        # apply a fixed torque
        self.torque = torque
        
    def compute_action(self, state):
        x, y, v = state
        
        if (x < 0): # because it is related to the theta
            action = np.sign(v)*self.torque # same direction to angular velocity
        else:
            action = (-1)*np.sign(v)*self.torque # opposite direction to angular velocity
        return action