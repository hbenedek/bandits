import numpy as np
import matplotlib.pyplot as plt

value_function = np.zeros(101)
policy = np.zeros(101)
states = np.arange(0,101)
rewards = np.zeros(101)
rewards[-1] = 1
gamma = 1
threshold = 0.0001

def value_iteration(p, max_sweep):
    """value iteration algo for solving the gamblers problem"""
    # initialization
    delta = 1
    
    # iterative update of the value function using the bellman equations
    sweep = 0
    while delta > threshold and sweep<max_sweep:
        delta = 0
        #one sweep over the state space
        #print(f'calculating sweep {sweep}')
        for state in states:
            current_value = value_function[state]
            value_function[state] = update_policy(state, p)
            delta = max(delta, np.abs(current_value - value_function[state]))
        sweep = sweep + 1

    # compute a deterministic policy using the value function
    #optimal_policy = compute_optimal_policy(value_function, gamma, p, rewards)

    return policy, value_function

def compute_expected_return_for_state_action(state, action, p):
    """computes the expected return for a state action pair using the Bellman equations"""   
    return p * (rewards[state + action] + gamma * value_function[state + action]) + (1 - p) * (0 + gamma * value_function[state - action])

def update_policy(state, p):
    """evaluating the expected return for each action, and choose greedily"""
    expected_returns = []
    actions = np.arange(0, min(state, 100 - state) + 1)
    for action in actions:
        expected_returns.append(compute_expected_return_for_state_action(state, action, p))
    policy[state] = np.argmax(expected_returns)
    return max(expected_returns)


        
if __name__=="__main__":
    policy, value = value_iteration(0.4, 1000)
    print(policy)
    plt.plot(policy)
    plt.show()

