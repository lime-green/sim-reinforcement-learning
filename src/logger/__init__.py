import os
from logging import getLogger


level = os.environ.get("LOG_LEVEL", "WARNING")
logger = getLogger("sim-reinforcement-learning")
logger.setLevel(level)
