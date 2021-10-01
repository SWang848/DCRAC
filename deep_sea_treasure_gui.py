"""Deep sea treasure envirnment

mostly compatible with OpenAI gym envs
"""

from __future__ import print_function

import sys
import math
import random
import itertools

import numpy as np
import scipy.stats
import gym

import pygame
print("Using pygame backend", file=sys.stderr)

from utils import truncated_mean, compute_angle, pareto_filter

SEA_MAP = [ 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [18, 26, 31, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [-1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [-1, -1, -1, 44, 48.2, 0, 0, 0, 0, 0, 0, 0],
    [-1, -1, -1, -1, -1, 56, 0, 0, 0, 0, 0, 0],
    [-1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0],
    [-1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0],
    [-1, -1, -1, -1, -1, -1, 72, 76.3, 0, 0, 0, 0],
    [-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0],
    [-1, -1, -1, -1, -1, -1, -1, -1, 90, 0, 0, 0],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, 100, 0, 0]]

HOME_POS = (0, 0)
SIZE_X = 12
SIZE_Y = 12

ACT_LEFT = 0
ACT_RIGHT = 1
ACT_UP = 2
ACT_DOWN = 3

ACTIONS = ["Left", "Right", "Up", "Down"]
ACTION_COUNT = len(ACTIONS)

# State space (whole map) vars
WIDTH = 480
HEIGHT = 480

BOX_WIDTH = int(WIDTH / SIZE_X)
BOX_HEIGHT = int(HEIGHT / SIZE_Y)

# Observation space vars
OBSERVE_WIDTH = int(WIDTH * 5 / 12)
OBSERVE_HEIGHT = int(HEIGHT * 5 / 12)

# Pre-calculated space vars for add margin
M_WIDTH = WIDTH + OBSERVE_WIDTH
M_HEIGHT = HEIGHT + OBSERVE_HEIGHT
W_MIN = int(OBSERVE_WIDTH/2)
W_MAX = int(OBSERVE_WIDTH/2 + WIDTH)
H_MIN = int(OBSERVE_HEIGHT/2)
H_MAX = int(OBSERVE_HEIGHT/2 + HEIGHT)

# Color definitions
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 51, 204)
YELLOW = (255, 212, 84)
GREY = (80, 80, 80)

FPS = 180

class DeepSeaTreasure():
    """Deep sea treasure environment
    """

    def __init__(self, sea_map=SEA_MAP):

        self.action_space = np.arange(ACTION_COUNT)
        self.observation_space = np.zeros((OBSERVE_WIDTH, OBSERVE_HEIGHT, 3), dtype=np.uint8)

        # Initialize graphics backend
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()

        # self.submarine_sprite = pygame.sprite.Sprite()
        # self.submarine_sprites = pygame.sprite.Group()
        # self.submarine_sprites.add(self.submarine_sprite)

        self.submarine_pos = list(HOME_POS)
        self.sea_map = sea_map

        self.end = False
        self.obj_cnt = 2

    def step(self, action, frame_skip=1, incremental_frame_skip=True):
        """Perform the given action `frame_skip` times
         ["Left", "Right", "Up", "Down"]
        Arguments:
            action {int} -- Action to perform, ACT_MINE (0), ACT_LEFT (1), ACT_RIGHT (2), ACT_ACCEL (3), ACT_BRAKE (4) or ACT_NONE (5)

        Keyword Arguments:
            frame_skip {int} -- Repeat the action this many times (default: {1})
            incremental_frame_skip {bool} -- If True, frame_skip actions are performed in succession, otherwise the repeated actions are performed simultaneously (e.g., 4 accelerations are performed and then the cart moves).

        Returns:
            tuple -- (observation, reward, terminal, info) tuple
        """

        reward = np.zeros(self.obj_cnt)
        if frame_skip < 1:
            frame_skip = 1

        reward[-1] = -1

        move = (0, 0)
        if action == ACT_LEFT:
            move = (-1,0)
        elif action == ACT_RIGHT:
            move = (1,0)
        elif action == ACT_UP:
            move = (0,-1)
        elif action == ACT_DOWN:
            move = (0,1)
        
        changed = self.move_submarine(move)
        reward[0] = self.sea_map[self.submarine_pos[1]][self.submarine_pos[0]]

        if not self.end and changed:
            self.render()

        info = self.get_state(True)
        observation = self.observation_space
        # observation (pixels), reward (list), done (boolean), info (dict)
        return observation, reward, self.end, info

    def move_submarine(self, move):
        target_x = self.submarine_pos[0] + move[0]
        target_y = self.submarine_pos[1] + move[1]
        
        if 0 <= target_x < 12 and 0 <= target_y < 12:
            if self.sea_map[target_y][target_x] != -1:
                self.submarine_pos[0] = target_x
                self.submarine_pos[1] = target_y
            
            if self.sea_map[target_y][target_x] != 0:
                self.end = True
            return True
        return False

    def get_pixels(self, update=True):
        """Get the environment's image representation

        Keyword Arguments:
            update {bool} -- Whether to redraw the environment (default: {True})

        Returns:
            np.array -- array of pixels, with shape (width, height, channels)
        """

        if update:
            self.pixels = pygame.surfarray.array3d(self.screen)
            
            self.get_observation()

        return self.pixels

    def get_state(self, update=True):
        """Returns the environment's full state, including the cart's position,
        its speed, its orientation and its content, as well as the environment's
        pixels

        Keyword Arguments:
            update {bool} -- Whether to update the representation (default: {True})

        Returns:
            dict -- dict containing the aforementioned elements
        """

        return {
            "position": self.submarine_pos,
            "pixels": self.get_pixels(update)
        }

    def get_observation(self):
        """Create a partially observable observation with the given state. 
        Half size of state["pixels"] with origin of state["position"], and the 
        overflow part is black
        
        Returns:
            array: 3d array as the same type of state["pixels"]
        """

        # original state pixels with margins
        margin_state_pixels = np.full((M_WIDTH, M_HEIGHT, 3), GREY[0], dtype=np.uint8)
        margin_state_pixels[W_MIN : W_MAX, H_MIN : H_MAX, :] = self.pixels

        min_x = int((self.submarine_pos[0]+0.5)*BOX_WIDTH)
        min_y = int((self.submarine_pos[1]+0.5)*BOX_HEIGHT)

        self.observation_space = margin_state_pixels[min_x : int(min_x + OBSERVE_WIDTH), min_y : int(min_y + OBSERVE_HEIGHT), :]
        return self.observation_space

    def reset(self):
        """Resets the environment to the start state

        Returns:
            [type] -- [description]
        """

        self.end = False
        self.submarine_pos = list(HOME_POS)
        self.render()
        self.get_state()
        return self.observation_space

    def __str__(self):
        string = "Completed: {} ".format(self.end)
        string += "Position: {} ".format(self.submarine_pos)
        return string

    def render(self):
        """Update the environment's representation
        """

        self.render_pygame()

    def render_pygame(self):

        pygame.event.get()
        self.clock.tick(FPS)

        # Clear canvas
        self.screen.fill(BLACK)

        # Draw Background
        for y in range(SIZE_Y):
            for x in range(SIZE_X):
                if self.sea_map[y][x] != -1:
                    r = pygame.Rect(x*BOX_WIDTH + 1, y*BOX_HEIGHT + 1, BOX_WIDTH - 2, BOX_HEIGHT - 2)
                    if self.sea_map[y][x] == 0:
                        pygame.draw.rect(self.screen, BLUE, r)
                    else:
                        pygame.draw.rect(self.screen, YELLOW, r)

        # Draw Submarine
        pygame.draw.circle(
            self.screen, 
            WHITE, 
            (int((self.submarine_pos[0]+0.5)*BOX_WIDTH), int((self.submarine_pos[1]+0.5)*BOX_HEIGHT)), 
            int(BOX_WIDTH/2 - 5))

        pygame.display.update()



# if __name__ == '__main__':
#     KEY = {pygame.K_LEFT: 0, pygame.K_RIGHT: 1, pygame.K_UP: 2, pygame.K_DOWN: 3, }

#     env = DeepSeaTreasure()
#     o_t = env.reset()
#     terminal = False
#     a_t = -1

#     while not terminal:
        
#         event = pygame.event.wait()
#         if event.type == pygame.KEYDOWN:
#             if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
#                 break
#             else:
#                 if event.key in KEY:
#                     a_t = KEY[event.key]
#                 else:
#                     continue
#             o_t1, r_t, terminal, s_t1 = env.step(a_t)
#             o_t = o_t1
#             print("Taking action", a_t, "at", s_t1["position"], "with reward", r_t)
#     env.reset()