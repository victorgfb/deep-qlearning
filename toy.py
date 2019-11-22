from enums import *
import random
from copy import deepcopy
from random import randrange

class ToySimulator:
    def __init__(self, length=6, small=2, large=10):
        self.length = length # Length of the dungeon
        #self.slip = slip  # probability of 'slipping' an action
        self.small = small  # payout for BACKWARD action
        self.large = large  # payout at end of chain for FORWARD action
        self.state = [0,0]  # Start at beginning of the dungeon
    
    def diff(self, newState, oldState):
        if(newState[0] > oldState[0] or newState[1] > oldState[1]):
            return True
        return False

    # def outOfArea(self, state):
    #     if()

    def take_action(self, action):
        # if random.random() < self.slip:
        #     action = not action  # agent slipped, reverse action taken
        
        self.oldState = deepcopy(self.state)

        if action == DOWN:  # BACKWARD: go back to the beginning, get small reward
            self.state[0]+=1
        elif action == UP:  # FORWARD: go up along the dungeon
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

        return self.state, reward

    def reset(self):
        self.state = [0,0] # Reset state to zero, the beginning of dungeon
        return self.state