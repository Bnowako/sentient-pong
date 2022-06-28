from collections import deque
from props import *

class Memory():
    def __init__(self):
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    def is_too_big(self):
        return len(self.memory) > BATCH_SIZE