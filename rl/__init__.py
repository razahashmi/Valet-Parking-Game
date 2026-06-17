"""RL scaffolding for the Valet-Park game."""
from rl.valet_env import ValetParkEnv

try:
    import gymnasium as gym
    from gymnasium.envs.registration import register

    if "ValetPark-v0" not in gym.registry:
        register(id="ValetPark-v0", entry_point="rl.valet_env:ValetParkEnv")
except Exception:
    pass

__all__ = ["ValetParkEnv"]
