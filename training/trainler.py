from stable_baselines3.common.vec_env  import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.vec_env  import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env  import DummyVecEnv
from stable_baselines3                 import PPO
from gymnasium                         import spaces
from config.config_base import ConfigBase
from glob import glob

import torch as th
import numpy as np
import random
import time
import os



class Trainler:
    def __init__(self, config : ConfigBase) -> None:
        self.config = config

    def set_seed(self,):
        random.seed(self.config.seed)
        os.environ['PYTHONHASHSEED'] = str(self.config.seed)
        th.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

    def make_env(self, env_class, env_config : ConfigBase):
        def _init():
            env = env_class(env_config.IMAGE_SIZE, env_config.LAST_N_FRAME)
            return env
        return _init

    def get_env(self):
        _env = make_vec_env(self.make_env(self.config.ENV, self.config), n_envs=1, seed=self.config.seed)
        _env = VecMonitor(_env)
        if ( os.path.isfile(self.config.VEC_NORM_PATH)):
            print(f"Loading VecNormalize from {self.config.VEC_NORM_PATH}")
            _env = VecNormalize.load(self.config.VEC_NORM_PATH, _env)
        else:
            print(f"Creating new VecNormalize at {self.config.VEC_NORM_PATH}")
            _env = VecNormalize(_env, 
                                norm_obs   = self.config.NORM_OBS, 
                                norm_reward= self.config.NORM_REWARD, 
                                clip_obs   = 10.)
        return _env
    
    def get_last_model(self):
        model_files = glob(f"{self.config.MODEL_DIR}/*.zip")
        if len(model_files) == 0:
            return None
        model_files.sort(key = os.path.getmtime)
        return model_files[-1]

    def build_model(self, env = None):
        if env is None:
            env = self.get_env()
        model_path = self.get_last_model()
        if model_path is not None:
            print(f"Loading model from {model_path}")
            _model = PPO.load(
                path            = model_path,
                env             = env,
                gamma           = self.config.gamma,
                n_steps         = self.config.N_STEPS,
                tensorboard_log = self.config.LOG_DIR,
                ent_coef        = self.config.ENT_COEF,
                seed            = self.config.seed,
                learning_rate   = self.config.lr,
                batch_size      = self.config.batch_size,
                target_kl       =0.01,
                verbose         = 1,
            )
        else:
            print("Creating new model")
            _model = PPO(
                policy          = self.config.MODEL,
                env             = env,
                gamma           = self.config.gamma,
                n_steps         = self.config.N_STEPS,
                tensorboard_log = self.config.LOG_DIR,
                ent_coef        = self.config.ENT_COEF,
                seed            = self.config.seed,
                learning_rate   = self.config.lr,
                batch_size      = self.config.batch_size,
                target_kl       =0.01,
                verbose         = 1,
            )
        return _model
    
    def learn(self):
        self.iters = 0
        model = self.build_model() # it will load model and auto update iter
        while True:
            self.iters += 1
            model.learn(total_timesteps     = self.config.TIME_STEPS,
                        tb_log_name         = self.config.TB_LOG_NAME,
                        reset_num_timesteps = False,  
                        )
            print(f"[*] Model will save to {self.config.MODEL_DIR}/{ int(self.config.TIME_STEPS*self.iters) }")
            model.save(f"{self.config.MODEL_DIR}/{ int(self.config.TIME_STEPS*self.iters) }.zip")
            model.env.save(self.config.VEC_NORM_PATH)

    def test_model(self):
        model = self.build_model()
        env   = model.get_env()
        while True:
            obs = env.reset()
            while True:
                action, _states = model.predict(obs)
                obs, reward, done, info = env.step(action)
                if done:
                    break
        