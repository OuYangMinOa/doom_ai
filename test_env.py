# testing the gymnasium environment
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


import time

from game.envs.game_screen import GamePage
from game.envs.en1 import DoomCaptchaEnv

def main():
    game_page = GamePage()
    game_page.keep_update_state()
    while 1:
        start = time.time()
        result = game_page.get_game_screen_image()
        print(result.shape)
        game_page.show_game_screen((240,160))
        fps  = 1/(time.time() - start)
        print(f"FPS: {fps}")
    

def test_env():
    env = DoomCaptchaEnv({})
    env.reset()

    for _ in range(1000):  # Run for 1000 steps
        # Sample a random action from the action space
        action = env.action_space.sample()
        # Take the action and observe the new state and reward
        observation, reward, done, info = env.step(action)
        if done:  # If the episode is done, reset the environment
            observation = env.reset()
    env.close()
    


if __name__ == "__main__":
    main()