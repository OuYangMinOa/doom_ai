import time
import numpy as np
import gymnasium as gym

from gymnasium import spaces
from collections import deque
from .game_screen import GamePage, GameState

class DoomCaptchaEnv(gym.Env):
    def __init__(self, image_size=(84, 84), last_n_frame=4):
        self.game = GamePage()
        self.action_space = spaces.Discrete(6) # 0: up, 1: down, 2: left, 3: right, 4: shoot, 5: do nothing
        self.observation_space = spaces.Box(low=0, high=255, shape=(image_size[0] * image_size[1] * 3 * last_n_frame + 1,), dtype=np.uint8)
        self.nlast_frame = deque(maxlen=last_n_frame)
        self.done = False
        self.reward = 0
        self.alive_time = 0

        self.image_w = image_size[0]
        self.image_h = image_size[1]
        self.last_n_frame = last_n_frame

        self.image_size = image_size

    def get_observation(self) -> np.ndarray:
        this_image = self.game.get_game_screen_image(self.image_size).flatten()
        self.nlast_frame.append(this_image)
        this_img_obs = np.zeros((self.last_n_frame, self.image_w*self.image_h*3), dtype=np.uint8)
        for i, frame in enumerate(self.nlast_frame):
            this_img_obs[i] = frame
        output = [
            this_img_obs.flatten(),
            np.array([self.score], dtype=np.uint8).flatten(),
        ]
        return np.concatenate(output).astype(np.uint8)

    def step(self, action) -> tuple:
        self.game.action(action)
        self.score  = self.game.get_current_score()
        self.state  = self.get_observation()
        self.reward = 0
        if self.game.current_state == GameState.DIED:
            self.done = True
            self.reward = -100
        elif self.game.current_state == GameState.WIN:
            self.reward =  100
            self.done = True
        else:
            self.reward = -1 * self.alive_time + 30 * self.score
        return self.state, self.reward, self.done, False, {"score": self.score, "state": self.game.current_state}


    def reset(self,seed=None, **kwargs) -> np.ndarray:
        super().reset(**kwargs)
        if (seed):
            np.random.seed(seed)
        self.done = False
        self.reward = 0
        self.alive_time = 0
        self.nlast_frame.clear()
        self.score = 0
        self.game.reset()
        return self.get_observation(), {}

    def render(self):
        pass

    def close(self):
        self.game.driver.quit()