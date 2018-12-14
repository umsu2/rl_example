import numpy as np
import math
import gym
import random as rand


def main():
    env = gym.make("Taxi-v2")
    learning_rate = 0.7
    discount_rate = 0.618
    total_training_episode = 5000
    max_step = 99
    qtable = QLearning(learning_rate, discount_rate, total_training_episode, max_step, env)
    run_agaist_simulation(10,qtable,max_step,env)
    env.stop()


# 5 steps to Q Learning

"""
1. Initialize Q Values
// also depends on the exploitation/ exploration 
2. Choose an action a in the current world state based on q value
3. take action and observe outcome
4. update Q values
5. repeat 

"""


def epsilon_generator(max, min, rate, step):
    return min + (max - min) * math.exp(-rate * step)


def should_explore(step):
    max = 1
    min = 0.01
    rate = 0.01
    epsilon = epsilon_generator(max, min, rate, step)
    r = rand.random()
    if r <= epsilon:
        return True


def initialize_Q_table(game):
    states = game.observation_space.n
    actions = game.action_space.n
    qtable = np.zeros((states, actions))
    return qtable


def QLearning(lr, dr, total_training_episodes, max_step, game):
    q_table = initialize_Q_table(game)
    for session in range(total_training_episodes):
        state = game.reset()
        for step in range(max_step):
            if should_explore(step):
                # take a random action from the action space
                action = game.action_space.sample()
            else:
                action = np.argmax(q_table[state, :])
            next_state, reward, done, info = game.step(action)
            # after action is taken, update the q table
            q_table[state, action] = q_table[state, action] + lr * (
                        reward + dr * np.max(q_table[next_state, :]) - q_table[state, action])
            state = next_state
            if done:
                break
    return q_table

def run_agaist_simulation(num_of_trials,q_table,max_step, game):

    for _ in range(num_of_trials):
        print("starting simulation")

        current_state = game.reset()
        reward_total = 0
        for _ in range(max_step):
            game.render()
            action = np.argmax(q_table[current_state,:])
            current_state, reward, done, info = game.step(action)
            reward_total += reward
            if done:
                break
        print("total reward for this game",reward_total)

if __name__ == "__main__":
    main()
