import gym
import numpy as np


def perfect_reward(reward):
    return reward


def static_variance_gaussian(reward):
    beta = 1
    beta0 = 1
    sigma = 1
    return static_variance_gaussian_(reward, beta, beta0, sigma)


def static_variance_gaussian_(reward, beta, beta0, sigma):
    return beta * reward + beta0 + np.random.normal(0, sigma)


def dynamic_variance_gaussian(reward, beta, beta0, t, sigma_f):
    sigma_t = sigma_f(t)
    return static_variance_gaussian(reward, beta, beta0, sigma_t)


def linearly_increasing_sigma(t):
    return 1e-3 * t


def solve_taxi_v2(n_episodes, gamma=0.99, alpha=0.5, epsilon=0.25, corruption_f=perfect_reward):
    env = gym.make('Taxi-v2')

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    q = np.random.random([n_states, n_actions])
    cumulative_rewards = []
    for episode in range(n_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = np.argmax(q[state, :])

            # explore at random
            e = np.random.random()
            if e < epsilon:
                action = env.action_space.sample()

            new_state, true_reward, done, info = env.step(action)
            corrupt_reward = corruption_f(true_reward)
            total_reward += true_reward

            old_q = q[state, action]
            new_q = corrupt_reward + gamma * np.max([q[new_state, new_action] for new_action in range(n_actions)])

            q[state, action] = alpha * new_q + (1 - alpha) * old_q

            state = new_state
        cumulative_rewards.append(total_reward)

    return q, cumulative_rewards


def play_taxi_v2(q, render_yes=False):
    env = gym.make('Taxi-v2')

    # play
    state = env.reset()
    if render_yes:
        env.render()
    done = False
    while not done:
        action = np.argmax(q[state, :])
        new_state, reward, done, info = env.step(action)
        state = new_state
        if render_yes:
            env.render()


