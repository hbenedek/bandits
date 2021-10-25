import numpy as np
import random

def value_iteration(threshold, gamma):
    # initialization
    delta = 100
    value_function = np.zeros(100)
    states = np.arange(1,100)
    rewards = np.zeros(100)
    # termination states
    rewards[0] = -1 
    rewards[99] = 1

    # iterative update of the value function using the bellman equations
    while delta > threshold:
        #one sweep over the state space
        for state in states:
            current_value = value_function[state]
            value_function[state] = update_policy()
            delta = max(delta, np.abs(current_value - value_function[state]))

    # compute a deterministic policy using the value function
    optimal_policy = compute_optimal_policy(gamma, value_function)

    return optimal_policy


def update_policy():
    # evaluating the expected return for each action, and choose greedily
    pass

def compute_optimal_policy(gamma, value_function):
    pass



def take_action(action, state, p):
    if action < 0 or action > min(state, 100 - state):
        raise ValueError('Opps, invalid action')
    coin_flip(p)

    
def coin_flip(p):
    return 1 if random.random()>p else -1

p = 0.4

#initialize states and rewards
states = np.arange(1,100)


#initialize value function

