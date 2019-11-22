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
        #print("----------")
        #print(state)
        dungeon = ToySimulator()
        while(state[0] != 5 or state[1] != 5):
            step+=1
            action = agent.greedy_action(deepcopy(state))
            #print(action)
            new_state, _ = dungeon.take_action(action)
            print(state)
            print(agent.get_Q(state))
            if(new_state[0] >= 6) or new_state[1] >= 6 or new_state[0] < 0 or new_state[1] < 0:
                print("droga")
                return False
            state = deepcopy(new_state)
        print("AEEEEEE")
        #print(step)
        return True

def main():
    iteracoes = 100000
    discout =  0.95
    learning_rate = 0.001

    agent = DeepQlearning(learning_rate=learning_rate, discount=discout, iterations=iteracoes)

    # setup simulation
    dungeon = ToySimulator()
    total_reward = 0 # Score keeping
    last_total = 0
    step = 0
    sucessCount = 0
    failCount = 0
    count = 0

    # main loop
    while(count < iteracoes):

        old_state = deepcopy(dungeon.state) 
        action = agent.get_next_action(old_state) 
        new_state, reward = dungeon.take_action(action)
        #print(new_state)
        if(new_state[0] == 5 and new_state[1] == 5):
            print("sucess")
            sucessCount += 1
            count += 1
            dungeon.reset()

        result = agent.update(deepcopy(old_state), deepcopy(new_state), action, reward) 
        total_reward += reward 
        
        if(not result):
            failCount+=1
            dungeon.reset()
            
        count += 1
        if((count % 250) == 0):
            performance = (total_reward - last_total) / 250
            print(json.dumps({'step': step, "performance" : performance, 'total_reward': total_reward}))
            print(sucessCount)
            sucessCount = 0
            last_total = total_reward
            total_reward = 0
        if(fit(agent)):
            print("olha só")
            fit(agent)
            break

if __name__ == "__main__":
    main()
