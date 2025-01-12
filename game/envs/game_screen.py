from selenium.webdriver.common.by import By
from selenium import webdriver
from pynput.keyboard import Key, Controller
from PIL import Image
from enum import Enum
from mss import mss


import io
import cv2
import time
import selenium
import threading
import numpy as np

class GameState:
    START = "Player born"
    DIED  = "Player died"
    WIN   = "Player win"

class GamePage:
    URL = "https://doom-captcha.vercel.app/"
    DRIVER_PATH = "chromedriver.exe"
    GAME_X = 117
    GAME_Y = 454
    GAME_H = 480
    GAME_W = 300
    def __init__(self):
        self.driver : webdriver.Chrome = webdriver.Chrome()
        self.driver.get(self.URL)
        self.driver.maximize_window()
        self.keyboard = Controller()
        self.game_screen  = self.driver.find_element(By.XPATH, "/html/body/div[4]/div[2]")
        time.sleep(2)
        self.open_debug_log()
        self.click_game_screen()
        self.current_state = GameState.START
        self.score = 0

    def reset(self):
        # refresh the page
        self.driver.refresh()
        time.sleep(2)
        self.score = 0
        self.current_state = GameState.START
        self.open_debug_log()
        self.game_screen  = self.driver.find_element(By.XPATH, "/html/body/div[4]/div[2]")
        self.click_game_screen()

    @property
    def game_screen_xywh(self) -> tuple[int, int, int, int]:
        return ( self.GAME_X,  self.GAME_Y,  self.GAME_H, self.GAME_W)

    def open_debug_log(self):
        self.driver.find_element(By.XPATH, f"/html/body/label/input").click()

    def click_game_screen(self):
        game_screen = self.driver.find_element(By.CLASS_NAME, "captcha-header")
        game_screen.click()

    def get_current_score(self) -> int:
        try:
            score = self.driver.find_element(By.XPATH, "/html/body/div[4]/div[3]/button").text[0]
            self.score = int(score)
            return self.score
        except Exception as e:
            print(e)
            self.score = 3
            self.current_state = GameState.WIN
            return self.score
        
    def keep_update_state(self):
        threading.Thread(target=self._update_state).start()

    def _update_state(self):
        while 1:
            self.current_state = self.get_current_state()

    def get_current_state(self) -> GameState:
        while 1:
            debug_info = self.driver.find_element(By.ID, "debug-log")
            last_line = debug_info.get_attribute("value").strip().split("\n")[-1]
            if "Player died" in last_line:
                return GameState.DIED
            elif "Player born" in last_line:
                return GameState.START
        
    def action(self, action: int):
        # action
        # 0: up
        # 1: down
        # 2: left
        # 3: right
        # 4: space
        if action == 0:
            self.keyboard.press(Key.up)
        elif action == 1:
            self.keyboard.press(Key.down)
        elif action == 2:
            self.keyboard.press(Key.left)
        elif action == 3:
            self.keyboard.press(Key.right)
        elif action == 4:
            self.keyboard.press(Key.space)

    def get_game_screen_image(self, resize : tuple = (480, 300)) -> np.ndarray:
        # get the game screen image
        x, y, w, h = self.game_screen_xywh
        with mss() as sct:
            screen = sct.grab({"left": x, "top": y, "width": w, "height": h} )
            img = Image.frombytes("RGB", screen.size, screen.bgra, "raw", "BGRX")
            if resize == (self.GAME_H, self.GAME_W):
                return np.array(img)
            else:
                img = img.resize(resize)
                return np.array(img)
    
    def show_game_screen(self, resize):
        img = self.get_game_screen_image(resize)
        cv2.imshow("Game Screen", img)
        cv2.waitKey(1)