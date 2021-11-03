import numpy as np
import random

def value_iteration(threshold, gamma, p):
    """value iteration algo for solving the gamblers problem"""
    # initialization
    delta = 100
    value_function = np.zeros(100)
    states = np.arange(1,100)
    rewards = np.zeros(100)
    # termination states
    rewards[0] = -1 
    rewards[99] = 1

    # iterative update of the value function using the bellman equations
    sweep = 0
    while delta > threshold:
        #one sweep over the state space
        print(f'calculating sweep {sweep}')
        for state in states:
            
            current_value = value_function[state]
            value_function[state] = update_policy(value_function, state, gamma, p)
            delta = max(delta, np.abs(current_value - value_function[state]))
        sweep += 1

    # compute a deterministic policy using the value function
    optimal_policy = compute_optimal_policy(gamma, value_function)

    return optimal_policy


def update_policy(value_function, state, gamma, p):
    """evaluating the expected return for each action, and choose greedily"""
    expected_returns = []
    actions = np.arange(0, min(state, 100 - state) + 1)
    for action in actions:
        expected_returns.append(compute_expected_return_for_state_action(state, action, gamma, p, value_function))
    return max(expected_returns)

def compute_optimal_policy(value_function, gamma, p):
    """returns a vector of the optimal policy, for a given state the vector contains the optimal value (value to bet)"""
    policy = np.zeros(100)
    for state in range(len(policy)):
        expected_returns = []
        actions = np.arange(0, min(state, 100 - state) + 1)
        for action in actions:
            expected_returns.append(compute_expected_return_for_state_action(state, action, gamma, p, value_function))
        optimal_action = np.argmax(expected_returns)
        policy[state] = optimal_action
    return policy
        
def compute_expected_return_for_state_action(state, action, gamma, p, value_function):
    """computes the expected return for a state action pair using the Bellman equations"""
    winning_destination_state = state + 2 * action 
    losing_destination_state = state - action 
    r = 1 if winning_destination_state == 100 else r = 0

    return p * (r + gamma * value_function[winning_destination_state]) + (1 - p) * (0 + gamma * value_function[losing_destination_state])

