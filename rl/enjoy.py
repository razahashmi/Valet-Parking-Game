"""Watch (or evaluate) a trained ValetParkEnv agent.

Examples (run from the repo root):
    python -m rl.enjoy rl/runs/valet_ppo_final.zip
    python -m rl.enjoy rl/runs/best/best_model.zip --episodes 5
    python -m rl.enjoy rl/runs/best/best_model.zip --render none   # headless stats only

Pass the same environment flags you trained with (--num-clients, --arrival-prob, ...).
"""
import argparse

import pygame
from stable_baselines3 import PPO

from rl.train import add_env_args, env_kwargs
from rl.valet_env import ValetParkEnv


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", help="path to a saved PPO .zip")
    p.add_argument("--episodes", type=int, default=3)
    add_env_args(p)  # same flags and defaults as rl.train
    p.add_argument("--render", choices=["human", "none"], default="human")
    p.add_argument("--deterministic", action="store_true",
                   help="take the greedy action instead of sampling (as in training); "
                        "greedy policies can lock up in this env")
    return p.parse_args()


def main():
    args = parse_args()
    render_mode = None if args.render == "none" else "human"
    env = ValetParkEnv(**env_kwargs(args, render_mode=render_mode))
    model = PPO.load(args.model)
    print(f"loaded {args.model}")

    wins = 0
    for ep in range(args.episodes):
        obs, info = env.reset(seed=ep)
        terminated = truncated = False
        total = 0.0
        quit_requested = False
        while not (terminated or truncated):
            if render_mode == "human":
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        quit_requested = True
                if quit_requested:
                    break
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total += reward
        wins += int(info["won"])  # time-outs also terminate, so check the outcome
        print(f"episode {ep}: return={total:8.2f}  "
              f"delivered={info['delivered_total']}/{env.total_clients}  "
              f"win={info['won']}")
        if quit_requested:
            break

    print(f"\nwins: {wins}/{args.episodes}")
    env.close()


if __name__ == "__main__":
    main()
