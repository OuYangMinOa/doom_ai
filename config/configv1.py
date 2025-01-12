from .config_base import ConfigBase

class config1(ConfigBase):
    TB_LOG_NAME   = "v1"
    MODEL_NAME    = "MODEL1"
    ENV_NAME      = "DOOM1"


class configv1_2(ConfigBase):
    TB_LOG_NAME   = "v1_2"
    MODEL_NAME    = "MODEL1"
    ENV_NAME      = "DOOM1"
    LAST_N_FRAME  = 3
    IMAGE_SIZE    = (240, 160)


class configv1_3(ConfigBase):
    TB_LOG_NAME   = "v1_3"
    MODEL_NAME    = "MODEL1"
    ENV_NAME      = "DOOM1"
    LAST_N_FRAME  = 3
    IMAGE_SIZE    = (240, 160)