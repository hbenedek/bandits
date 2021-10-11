from abc import abstractmethod
import statistics
import numpy as np
import matplotlib.pyplot as plt
import random

class Bandit:
    def __init__(self, arms: int):
        self.arms = arms

    @abstractmethod
    def take_action(self, action: int):
        pass

class SimpleBandit(Bandit):
    def __init__(self, mu: float, sigma: float):
        self.mu = mu
        self.sigma = sigma
        self.expected_rewards = statistics.NormalDist(mu, sigma).samples(self.arms)
        self.rewards = [statistics.NormalDist(mu, 1) for mu in self.expected_rewards]

    def take_action(self, action: int) -> float:
        return self.rewards[action].samples(1)[0]

class RandomWalkBandit(Bandit):
    def __init__(self, mu, sigma, arms):
        super().__init__(arms)
        self.mu = mu
        self.sigma = sigma
        self.initial_expected_rewards = statistics.NormalDist(mu, sigma).samples(self.arms)
        self.expected_reward_history = np.array([self.initial_expected_rewards])

    def get_current_expected_rewards(self) -> np.array:
        return self.expected_reward_history[-1]

    def random_step(self) -> np.array:
        return np.array([1 if random.random() < 0.5 else -1 for _ in range(self.arms)])

    def update_expected_rewards(self, step_size: float):
        new_rewards = self.get_current_expected_rewards + self.random_step() * step_size
        self.expected_reward_history = np.vstack([self.expected_reward_history, new_rewards])

    def take_action(self, action: int) -> float:
        return statistics.NormalDist(self.get_current_expected_rewards()[action], 1).samples(1)[0]



class Agent:
    def __init__(self, arms):
        self.arms = arms
        self.wealth_history = [0]
        self.wealth = 0
        self.average_wealth = [0]
        self.max_iter = None

    @abstractmethod
    def play(self, bandit: Bandit, max_iter: int):
        pass

    def update_wealth(self, reward: float, iter: int):
        self.wealth += reward
        self.wealth_history.append(self.wealth)
        self.average_wealth.append(self.average_wealth[-1] + 1 / (iter + 1) * (reward - self.average_wealth[-1]))

class EpsilonGreedyAgent(Agent):
    def __init__(self, arms, optimistic=False):
        super().__init__(arms)
        if optimistic:
            self.estimates = list(np.ones(self.arms) * 5)
        else:
            self.estimates = list(np.zeros(self.arms))
        self.action_history = np.zeros(self.arms)

    def update_estimate(self, reward: float, action: int):
        self.action_history[action] += 1 
        self.estimates[action] = self.estimates[action] + (1 / self.action_history[action]) * (reward - self.estimates[action])

    def policy(self, epsilon: float) -> int:
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
    def __init__(self, arms):
        super().__init__(arms)
        self.preferences = np.random.uniform(0, 1, self.arms)   

    def update_preference(self, action, reward, alpha): 
        updated = self.preferences[action] + alpha * (reward - self.average_wealth) * (1 - self.softmax(self.preferences)[action])
        self.preferences = self.preferences - alpha * (reward - self.average_wealth) * self.softmax(self.preferences)
        self.preferences[action] = updated
    
    def softmax(v):
        denominator = np.sum(np.exp(v))
        return v / denominator

    def policy(self):
        softmax = self.softmax(self.preferences)
        return self.preferences.index(max(softmax))

    def play(self, bandit: Bandit, max_iter: int):
        for iter in range(max_iter):
            action = self.policy(self.estimates)
            reward = bandit.take_action(action)
            self.update_wealth(reward, iter) 
            self.preferences = self.update_preference(action)
               

class UpperConfidanceIntervalAgent(Agent):
    def __init__(self, arms):
        super().__init__(arms)
        pass   



