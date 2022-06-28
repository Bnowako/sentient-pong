import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
from props import *
pygame.init()
font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

class Env:
    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.reset()

    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0


    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()


    def step(self, action):
        self.frame_iteration += 1
        
        self._move(action) 
        self.snake.insert(0, self.head)
        
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
        
        return reward, game_over, self.score


    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True

        return False

    def get_dangers(self):
        head = self.head
        block_on_left = Point(head.x - BLOCK_SIZE, head.y)
        block_on_right = Point(head.x + BLOCK_SIZE, head.y)
        space_on_up = Point(head.x, head.y - BLOCK_SIZE)
        block_on_down = Point(head.x, head.y + BLOCK_SIZE)

        dangers = [
            (self.is_dir_right() and self.is_collision(block_on_right)) or 
            (self.is_dir_left() and self.is_collision(block_on_left)) or 
            (self.is_dir_up() and self.is_collision(space_on_up)) or 
            (self.is_dir_down() and self.is_collision(block_on_down)),

            (self.is_dir_up() and self.is_collision(block_on_right)) or 
            (self.is_dir_down() and self.is_collision(block_on_left)) or 
            (self.is_dir_left() and self.is_collision(space_on_up)) or 
            (self.is_dir_right() and self.is_collision(block_on_down)),

            (self.is_dir_down() and self.is_collision(block_on_right)) or 
            (self.is_dir_up() and self.is_collision(block_on_left)) or 
            (self.is_dir_right() and self.is_collision(space_on_up)) or 
            (self.is_dir_left() and self.is_collision(block_on_down)),
            ]
        
        return dangers
        
    def get_directions_state(self):
        return [
            self.is_dir_up(),
            self.is_dir_right(),
            self.is_dir_down(),
            self.is_dir_left(),
        ]

    def is_dir_right(self):
        return self.direction == Direction.RIGHT

    def is_dir_left(self):
        return self.direction == Direction.LEFT

    def is_dir_up(self):
        return self.direction == Direction.UP
    
    def is_dir_down(self):
        return self.direction == Direction.DOWN

    def get_food_location(self):
        return [
            self.food.x < self.head.x,
            self.food.x > self.head.x, 
            self.food.y < self.head.y, 
            self.food.y > self.head.y 
            ]


    def draw(self, display):
        display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        display.blit(text, [0, 0])
        pygame.display.flip()


    def _move(self, action):

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)