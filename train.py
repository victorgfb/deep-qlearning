import random
import json
import argparse
import time
from deepQlearning import DeepQlearning
from toy import ToySimulator
from copy import deepcopy

def fit(agent):
    for _ in range(1):
        step = 0
        state = [0,0]
     
        dungeon = ToySimulator()
        while(state[0] != 5 or state[1] != 5):
            step+=1
            action = agent.greedy_action(deepcopy(state))
            #print(action)
            new_state, _ = dungeon.take_action(action)
            print(state)
            print(agent.get_Q(state))
            if(new_state[0] >= 6) or new_state[1] >= 6 or new_state[0] < 0 or new_state[1] < 0:
                print("ops")
                return False
            state = deepcopy(new_state)
        print("AEEEEEE")
        return True

def main():
    max_iteracoes = 1000000
    discout =  0.75
    learning_rate = 0.001

    agent = DeepQlearning(learning_rate=learning_rate, discount=discout, iterations=max_iteracoes)

    dungeon = ToySimulator()
    count = 0

    while(count < max_iteracoes):
        count+=1
        old_state = deepcopy(dungeon.state) 
        action = agent.get_next_action(old_state) 
        new_state, reward = dungeon.take_action(action)
        
        if(new_state[0] == 5 and new_state[1] == 5):
            print("sucess")
            dungeon.reset()

        result = agent.update(deepcopy(old_state), deepcopy(new_state), action, reward) 

        if(not result):
            dungeon.reset()
            
        if(fit(agent)):
            print("olha só")
            break

if __name__ == "__main__":
    main()
