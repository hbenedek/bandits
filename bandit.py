from abc import abstractmethod
import statistics
import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self, k, mu, sigma):
        self.arms = k

    @abstractmethod
    def take_action(self, action):
        pass

class SimpleBandit(Bandit):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self.expected_rewards = statistics.NormalDist(mu, sigma).samples(k)
        self.rewards = [statistics.NormalDist(mu, 1) for mu in self.expected_rewards]

    def take_action(self, action):
        return self.rewards[action].samples(1)[0]

class RandomWalkBandit(Bandit):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def take_action(self, action):
        pass



class Agent:
    def __init__(self, k):
        self.arms = k

class EpsilonGreedyAgent(Agent):
    def __init__(self):
        self.estimates = list(np.zeros(self.arms))
        self.action_history = np.zeros(self.arms)
        self.wealth_history = [0]
        self.wealth = 0
        self.average_wealth = [0]
        self.max_iter = None

    def update_estimate(self, reward, action):
        self.action_history[action] += 1 
        self.estimates[action] = self.estimates[action] + (1 / self.action_history[action]) * (reward - self.estimates[action])

    def update_wealth(self, reward, iter):
        self.wealth += reward
        self.wealth_history.append(self.wealth)
        self.average_wealth.append(self.average_wealth[-1] + 1 / (iter + 1) * (reward - self.average_wealth[-1]))

    def policy(self, estimates, epsilon):
        if np.random.rand() <= epsilon:
            return np.random.randint(0, self.arms)
        else:
            return self.estimates.index(max(self.estimates))
         
    
    def play(self, bandit: Bandit, max_iter: int, epsilon: float):
        self.max_iter = max_iter
        for iter in range(max_iter):
            action = self.policy(self.estimates, epsilon)
            reward = bandit.take_action(action)
            self.action_history[action] += 1
            self.update_estimate(reward, action)
            self.update_wealth(reward, iter)

    def plot_result(self):
        plt.plot(np.arange(self.max_iter + 1), self.average_wealth)

class GradientAgent(Agent):
    def __init__(self):
        pass    




