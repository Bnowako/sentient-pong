import torch
import random
import numpy as np
from collections import deque
from game import Env, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot
import os
import pygame
from props import *
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 256, 3)
        if os.path.exists(f'./model/model.pth'):
            self.model.load_state_dict(torch.load(f'./model/model.pth'))
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, env):
        is_going_left = env.direction == Direction.LEFT
        is_going_right = env.direction == Direction.RIGHT
        is_going_up = env.direction == Direction.UP
        is_going_down = env.direction == Direction.DOWN

        state =  env.get_dangers() + [
            # Move direction
            is_going_left,
            is_going_right,
            is_going_up,
            is_going_down,
            
            # Food location 
            env.food.x < env.head.x,  # food left
            env.food.x > env.head.x,  # food right
            env.food.y < env.head.y,  # food up
            env.food.y > env.head.y  # food down
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0

    agent = Agent()
    env = Env(640, 480)
    clock = pygame.time.Clock()
    
    if DRAW: 
        display = pygame.display.set_mode((640, 480))
    while True:
        state_old = agent.get_state(env)

        final_move = agent.get_action(state_old)

        reward, done, score = env.step(final_move)
        
        if DRAW:
            env.draw(display)
            clock.tick(SPEED)

        state_new = agent.get_state(env)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            env.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
        if agent.n_games == 100:
            print('Game', agent.n_games, 'Score', score, 'Record:', record, 'MEAN: ', total_score / agent.n_games)
            break


if __name__ == '__main__':
    train()