import torch
import random
from game import Env
from model import QNet
from trainer import Trainer
from charts import scores_chart, comparison_chart
import os
import pygame
from props import *
from memory import Memory
from pprint import pprint
import inquirer


class Agent:

    def __init__(self, gamma, epsilon, learning_rate):
        self.n_games = 0
        self.epsilon = epsilon
        self.gamma = gamma
        self.memory = Memory()
        if os.path.exists('snake_ai_model.pt'):
            print('train')
            self.model = torch.jit.load('snake_ai_model.pt')
            self.model.eval()
        else:
            self.model = QNet(11, 256, 3)
        print(self.model)
        self.trainer = Trainer(self.model, lr=learning_rate, gamma=self.gamma)

    def train_long_memory(self):
        if self.memory.is_too_big():
            mini_sample = random.sample(self.memory.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train(number_of_games, gamma, epsilon, learning_rate, train_from_beginning):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0

    agent = Agent(gamma, epsilon, learning_rate)
    if train_from_beginning:
        agent.model = QNet(11, 256, 3)
        agent.trainer = Trainer(agent.model, lr=learning_rate, gamma=agent.gamma)
        record = 0
    else:
        f = open('record.txt', 'r')
        record = int(f.read())
        f.close()

    print(agent.model)
    print('Record:')
    print(record)

    env = Env(640, 480)
    clock = pygame.time.Clock()

    if is_gui_visible:
        display = pygame.display.set_mode((640, 480))
    while True:
        state_old = env.get_state()

        final_move = agent.get_action(state_old)

        reward, done, score = env.step(final_move)

        if is_gui_visible:
            env.draw(display)
            clock.tick(SPEED)

        state_new = env.get_state()

        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.memory.remember(state_old, final_move, reward, state_new, done)

        if done:
            env.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                if not train_from_beginning:
                    f = open('record.txt', 'w')
                    f.write(str(record))
                    f.close()
                    f = open('record.txt', 'r')
                    record = int(f.read())
                    f.close()
                    model_scripted = torch.jit.script(agent.model)  # Export to TorchScript
                    model_scripted.save('snake_ai_model.pt')

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            scores_chart(plot_scores, plot_mean_scores)

        if agent.n_games == number_of_games:
            average_score = total_score / agent.n_games
            print('Game', agent.n_games, 'Score', score, 'Record:', record, 'MEAN: ', average_score)
            break

    return average_score


if __name__ == '__main__':
    run_model_type_questions = [
        inquirer.List(
            "run_model_type",
            message="How do you want to run our model?",
            choices=['watchTrainedModel', 'trainFromBeginning', 'compareModels'],
        ),
    ]

    answers = inquirer.prompt(run_model_type_questions)
    pprint(answers['run_model_type'])

    run_model_type = answers['run_model_type']

    display_gui_questions = [
        inquirer.List(
            "display_gui",
            message="Do you want to see snake gui? (training might take more time)",
            choices=['yes', 'no'],
        ),
    ]

    answers = inquirer.prompt(display_gui_questions)
    pprint(answers['display_gui'])

    display_gui = answers['display_gui']
    is_gui_visible = display_gui == 'yes'

    if run_model_type == 'watchTrainedModel':
        iterations = [
        inquirer.List(
            "iterations",
            message="Do you want to see snake gui? (training might take more time)",
            choices=['100','500', '1000'],
        ),
    ]
        answers = inquirer.prompt(iterations)
        print('watchTrainedModel')
        train(int(answers['iterations']), 0.9, 1, 0.001, False)

    if run_model_type == 'trainFromBeginning':
        print('trainFromBeginning')
        train(100, 0.9, 1, 0.001, True)
        train(100, 1, 3, 0.00025, True)
        train(100, 0.9, 1, 0.001, True)

    if run_model_type == 'compareModels':
        print('compareModels')
        trained_model_average_score = train(100, 0.9, 1, 0.001, False)
        not_trained_model_average_score = train(100, 0.9, 1, 0.001, True)
        comparison_chart(trained_model_average_score, not_trained_model_average_score)
