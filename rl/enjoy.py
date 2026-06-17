"""Watch (or evaluate) a trained ValetParkEnv agent.

Examples (run from the repo root):
    python -m rl.enjoy rl/runs/valet_ppo_final.zip
    python -m rl.enjoy rl/runs/best/best_model.zip --episodes 5
    python -m rl.enjoy rl/runs/best/best_model.zip --render none   # headless stats only
"""
import argparse

import pygame
from stable_baselines3 import PPO

from rl.valet_env import ValetParkEnv


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", help="path to a saved PPO .zip")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--game-time", type=int, default=60)
    p.add_argument("--render", choices=["human", "none"], default="human")
    p.add_argument("--stochastic", action="store_true",
                   help="sample actions instead of taking the greedy one")
    return p.parse_args()


def main():
    args = parse_args()
    render_mode = None if args.render == "none" else "human"
    env = ValetParkEnv(render_mode=render_mode, game_time=args.game_time)
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
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total += reward
        wins += int(terminated)
        print(f"episode {ep}: return={total:8.2f}  "
              f"delivered={info['delivered_total']}/{env.total_clients}  "
              f"win={terminated}")
        if quit_requested:
            break

    print(f"\nwins: {wins}/{args.episodes}")
    env.close()


if __name__ == "__main__":
    main()
