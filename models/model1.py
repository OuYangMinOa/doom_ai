from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from gymnasium import spaces
from torch import nn

import torch as th

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import PPO

from .utils.resnet import ResNet

class DOOM1(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super().__init__()  
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        self.activation = nn.ReLU

        self.cnn_increase = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=1, stride=1, padding="same"),
            )
        
        self.resnet = ResNet(input_dim=8, feature_dim=8, num_blocks=4, activation=self.activation)
        self.cnn_decrease = nn.Sequential(
            nn.Conv2d(8, 4, kernel_size=3, stride=1, padding="same"),
            nn.MaxPool2d(kernel_size=2, stride=2), # 240, 160 -> 120, 80
            self.activation(),
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding="same"),
            nn.MaxPool2d(kernel_size=2, stride=2), # 120, 80 -> 60, 40
            self.activation(),
            nn.Conv2d(4, 3, kernel_size=3, stride=1, padding="same"),
            nn.MaxPool2d(kernel_size=2, stride=2), # 60, 40 -> 30, 20
            self.activation(),  
        )

        self.cnn_to_linear = nn.Sequential(
            nn.Linear(3 * 30 * 20, 512),
            self.activation(),
            nn.Linear(512, 256),
            self.activation(),
            nn.Linear(256, 128),
            self.activation(),
        )

        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)

        self.atten_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 128, 256),
            self.activation(),
            nn.Linear(256, 128),
            self.activation(),
            nn.Linear(128, 128),
            self.activation(),
        )

        self.actor_dnn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 + 1, 64),
            self.activation(),
            nn.Linear(64, 64),
            self.activation(),
            nn.Linear(64, self.latent_dim_pi),
        )

        self.critic_dnn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 + 1, 64),
            self.activation(),
            nn.Linear(64, 64),
            self.activation(),
            nn.Linear(64, self.latent_dim_vf),
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)
    
    def forward_before_split(self, features: th.Tensor) -> th.Tensor:
        batch_size = features.shape[0]
        pic   = features[:, 0:-1].float() / 255.
        scroe = features[:, -2:-1].float() / 3.
        pic   = pic.reshape(batch_size * 3,240, 160, 3)
        pic   = pic.transpose(1, 3).transpose(2, 3) # 240, 160, 3 -> 3, 240, 160
        pic   = self.cnn_increase(pic) 
        pic   = self.resnet(pic)
        pic   = self.cnn_decrease(pic)
        pic   = pic.reshape(batch_size, 3, 3 * 30 * 20)
        pic   = self.cnn_to_linear(pic)
        pic,_ = self.attention(pic, pic, pic)
        pic   = self.atten_linear(pic)
        pic   = th.cat([pic, scroe], dim=1)
        return pic
    
    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.actor_dnn(self.forward_before_split(features))

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.critic_dnn(self.forward_before_split(features))



class model1(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = True
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = DOOM1(self.features_dim)