from codecs import backslashreplace_errors
import torch
import random
import numpy as np
from collections import deque
from model import Linear_QNet, QTrainer
from helper import plot
from pong import Game
import pygame
import matplotlib.pyplot as plt
import os
from datetime import datetime

from pong.game import GameInformation
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.00025

class Agent:

    def __init__(self, name):
        self.n_games = 0
        self.epsilon = 1 # randomness
        self.gamma = 0.99 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(3, 128, 2, name)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        if os.path.exists(f'./model/{name}model.pth'):
            self.model.load_state_dict(torch.load(f'./model/{name}model.pth'))
    
    def get_state(self, game, left):
        ball = game.ball
        left_paddle = game.left_paddle
        right_paddle = game.right_paddle
        
        paddle = left_paddle if left else right_paddle
        velocity = 1 if ball.x_vel < 0 else -1

        dif_x = abs(paddle.x - ball.x)
        dif_y = paddle.y - ball.y 
        state = [
            velocity,
            dif_y > 0,
            dif_x / game.window_width,
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 1)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move


def train_pong():
    plt_left_hits = []
    plt_mean_left_hits = []
    total_left_hits = []
    left_record = 0
    right_record = 0
    agent = Agent("left-")
    agent2 = Agent("right-")
    
    height = 500
    width = 700

    win = pygame.display.set_mode((width, height))
    game = Game(win, width, height)
    
    clock = pygame.time.Clock()

    run = True
    prev_game_info = GameInformation(0,0,0,0)
    done = False

    while run:
        clock.tick(1000)
        state_old = agent.get_state(game, True)
        state_old2 = agent2.get_state(game, False)

        final_move = agent.get_action(state_old)
        final_move2 = agent2.get_action(state_old2)

        game.play_predicted_move(True, final_move)
        ball, current_gem_left_hits = game.play_predicted_move(False, final_move2)
        # todo! is this correct game_info?
        game_info = game.loop()
        
        reward, reward2 = add_rewards(game_info, prev_game_info)
        
        state_new = agent.get_state(game, True)
        state_new2 = agent2.get_state(game, False)

        if game_info.right_score + game_info.left_score == 50 or game.current_gem_left_hits > 30:
            done = True
        else:
            done = False
        
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent2.train_short_memory(state_old2, final_move2, reward2, state_new2, done)

        agent.remember(state_old, final_move, reward, state_new, done)        
        agent2.remember(state_old2, final_move2, reward2, state_new2, done)        
        
        if(done):
            print('Game', agent.n_games, 'Left hits', game.left_hits,'Right hits',game.right_hits)
            print('Game', agent.n_games, 'Left score', game.left_score,'Right score',game.right_score, 'Record:', left_record)
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            agent2.train_long_memory()
            prev_game_info = GameInformation(0,0,0,0)

            # IF the score is better, save model
            if game_info.left_score > left_record:
                left_record = game_info.left_score
                agent.model.save()
            if game_info.right_score > right_record:
                right_record = game_info.right_score
                agent2.model.save()

            # plot_results(plt_left_scores, total_left_scores, plt_mean_left_scores, agent, game_info.left_score)
            plot_results(plt_left_hits, total_left_hits, plt_mean_left_hits, agent, game_info.left_hits)

        game.draw(draw_score=True, draw_hits=True)
        # pygame.display.update()

def play_with_human():
    record = 0
    agent = Agent("left-")

    height = 500
    width = 700

    win = pygame.display.set_mode((width, height))
    game = Game(win, width, height)
    
    clock = pygame.time.Clock()

    run = True
    prev_game_info = GameInformation(0,0,0,0)
    done = False
    while run:
        clock.tick(1000)
        
        state_old = agent.get_state(game, True)


        final_move = agent.get_action(state_old)

        game.play_predicted_move(True, final_move)
        game_info = game.loop()
        
        reward = add_rewards(game_info, prev_game_info,)
        
        state_new = agent.get_state(game, True)

        if game_info.right_score + game_info.left_score == 50:
            done = True
        else:
            done = False
        
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)    
        
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_w]:
            game.move_paddle(left=False, up=True)
        elif keys[pygame.K_s]:
            game.move_paddle(left=False, up=False)


        if(done):
            print('Game', agent.n_games,'Right hits',game.right_hits)
            print('Game', agent.n_games, 'Left score', game.left_score,'Right score',game.right_score, 'Record:', record)
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            prev_game_info = GameInformation(0,0,0,0)

            if game_info.left_score > record:
                record = game_info.left_score
                agent.model.save()

        game.draw(draw_score=True, draw_hits=True)
        pygame.display.update()
        pygame.event.pump()
    

def add_rewards(game_info, prev_game_info):
    #add reward to left_paddle model
    reward = 0
    if game_info.left_hits > prev_game_info.left_hits:
            reward += 1
    if game_info.right_score > prev_game_info.right_score:
        reward -= 1
    
    #add reward to right_paddle model
    reward2 = 0
    if game_info.right_hits > prev_game_info.right_hits:
        reward2 = 1
    if game_info.left_score > prev_game_info.left_score:
        reward2 -= 1
    
    prev_game_info.left_score = game_info.left_score
    prev_game_info.right_score = game_info.right_score
    prev_game_info.right_hits = game_info.right_hits
    prev_game_info.left_hits = game_info.left_hits


    # if(reward != 0):
        # print(f'time:{datetime.now()}:  {reward} ')
    return reward, reward2

def plot_results(plt_scores, all_scores, mean_scores, agent, value):
    plt_scores.append(value)
    all_scores.append(value)
    mean_score = sum(all_scores) / agent.n_games
    mean_scores.append(mean_score)
    plot(plt_scores, mean_scores)


if __name__ == '__main__':
    train_pong()
    # play_with_human()