from torch import nn
import torch as th

import os

from models.model1 import model1
from game.envs.en1 import DoomCaptchaEnv

class ConfigBase:
    seed          = 42
    lr            = 3e-4
    gamma         = 0.99
    batch_size    = 32
    buffer_size   = 10000
    DIR_HEAD      = "train_doom_data/"
    TB_LOG_NAME   = "v1"
    MODEL_NAME    = "MODEL1"
    ENV_NAME      = "DOOM1"
    ENV           = DoomCaptchaEnv
    MODEL         = model1
    N_STEPS       = 2048
    ENT_COEF      = 0.01
    TIME_STEPS    = 2000
    IMAGE_SIZE    = (240, 160)
    LAST_N_FRAME  = 10
    NORM_OBS      = True
    NORM_REWARD   = True
    TARGET_KL     = None
    CLIP_OBS      = 10.
    


    @property
    def MODEL_DIR(self):
        temp_model_dir = f"{self.DIR_HEAD}/models/{self.MODEL_NAME}/{self.ENV_NAME}/{self.TB_LOG_NAME}/"
        os.makedirs(temp_model_dir, exist_ok=True)
        return temp_model_dir

    @property
    def LOG_DIR(self):
        temp_log_dir = f"{self.DIR_HEAD}/logs/{self.MODEL_NAME}/{self.ENV_NAME}/"
        os.makedirs(temp_log_dir, exist_ok=True)
        return temp_log_dir
    
    @property
    def VEC_NORM_PATH(self):
        temp_log_dir = f"{self.DIR_HEAD}/vec/{self.MODEL_NAME}/{self.ENV_NAME}/"
        os.makedirs(temp_log_dir, exist_ok=True)
        FILENAME = f"{self.TB_LOG_NAME}.pkl"
        return f"{temp_log_dir}/{FILENAME}"