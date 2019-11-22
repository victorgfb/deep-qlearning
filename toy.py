from enums import *
import random
from copy import deepcopy
from random import randrange

class ToySimulator:
    def __init__(self, length=6, small=2, large=10):
        self.length = length
        self.small = small 
        self.large = large 
        self.state = [0,0]  

    def diff(self, newState, oldState):
        if(newState[0] > oldState[0] or newState[1] > oldState[1]):
            return True
        return False

    def take_action(self, action):
  
        self.oldState = deepcopy(self.state)

        if action == DOWN: 
            self.state[0]+=1
        elif action == UP:  
            self.state[0]-=1
        elif action == LEFT:
            self.state[1]-=1
        elif action == RIGHT:
            self.state[1]+=1


        if((self.state[0] >= self.length) or (self.state[0] < 0)):
            reward = -100
        elif ((self.state[1] >= self.length) or (self.state[1] < 0)):
            reward = -100
        else:
            if(self.diff(self.state, self.oldState)):
                reward = 10
            else:
                reward = 0

        # if(self.state[1] < 5 and self.state[0] == 1):
        #     reward = -100

        return self.state, reward

    def reset(self):
        self.state = [0,0] 
        return self.state