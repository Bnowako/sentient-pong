from codecs import backslashreplace_errors
import torch
import random
import numpy as np
from collections import deque
from model import Linear_QNet, QTrainer
from helper import plot
from pong import Game
import pygame
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self, name):
        self.n_games = 0
        self.epsilon = 1 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(4, 256, 3, name)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game, left):
        ball = game.ball
        left_paddle = game.left_paddle
        right_paddle = game.right_paddle
        
        paddle = left_paddle if left else right_paddle

        # ball_level = abs(ball.y - paddle.y) < 5
        velocity = 1 if ball.x_vel < 0 else -1

        # distance =  abs(ball.x - paddle.x) / game.window_width
        dif_x = left_paddle.x - ball.x
        dif_y = left_paddle.y - ball.y 

        state = [
            velocity,
            dif_y > 0,
            dif_y < 0,
            dif_x / game.window_width,
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
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
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

def train_pong():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent("left-")
    agent2 = Agent("right-")
    # game = SnakeGameAI()
    
    height = 500
    width = 700

    win = pygame.display.set_mode((width, height))
    game = Game(win, width, height)

    
    clock = pygame.time.Clock()

    run = True
    prev_left_hits = 0
    prev_right_score = 0
    prev_left_score = 0
    prev_right_hits = 0
    done = False
    while run:
        clock.tick(500)
        state_old = agent.get_state(game, True)
        state_old2 = agent2.get_state(game, False)

        final_move = agent.get_action(state_old)
        final_move2 = agent.get_action(state_old2)

        game.play_predicted_move(True, final_move)
        ball_y = game.play_predicted_move(False, final_move2)
        game_info = game.loop()
        
        
        if game_info.left_hits > prev_left_hits:
            reward = 20
            prev_left_hits = game_info.left_hits
        else:
            reward = 0
        
        if game_info.right_score > prev_right_score:
            if ball_y < game.left_paddle.y:
                if final_move[0] != 1:
                    reward -= 10
                if final_move[0] == 1:
                    reward += 10
            elif ball_y > game.left_paddle.y:
                if final_move[2] != 1:
                    reward -= 10
                else:
                    reward +=10
            
            prev_right_score = game_info.right_score
            

        
        
        if game_info.right_hits > prev_right_hits:
            reward2 = 15
            prev_right_hits = game_info.right_hits
        else :
            reward2 = 0

        state_new = agent.get_state(game, True)
        state_new2 = agent.get_state(game, False)


        if game_info.right_score + game_info.left_score == 20:
            done = True
        else:
            done = False
        
        if(reward != 0):
            print(f'reward: {reward}')
        
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent2.train_short_memory(state_old2, final_move2, reward2, state_new2, done)

        agent.remember(state_old, final_move, reward, state_new, done)        
        agent.remember(state_old2, final_move2, reward2, state_new2, done)        
        
        if(done):
            print('Game', agent.n_games, 'Left hits', game.left_hits,'Right hits',game.right_hits, 'Record:', record)
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            prev_left_hits = 0
            prev_right_hits = 0
            prev_left_score = 0
            prev_right_score = 0

            if game_info.left_hits > record:
                record = game_info.left_hits
                agent.model.save()


            plot_scores.append(game_info.left_hits)
            total_score += game_info.left_hits
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

        game.draw(draw_score=True, draw_hits=True)
        pygame.display.update()
    

if __name__ == '__main__':
    train_pong()