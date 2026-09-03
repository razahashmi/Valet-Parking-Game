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
    p.add_argument("--frame-skip", type=int, default=4, help="must match training")
    p.add_argument("--num-clients", type=int, default=10)
    p.add_argument("--max-cars", type=int, default=10, help="must match training (obs size)")
    p.add_argument("--arrival-prob", type=float, default=0.5, help="match training regime")
    p.add_argument("--exit-prob", type=float, default=0.5, help="match training regime")
    p.add_argument("--render", choices=["human", "none"], default="human")
    p.add_argument("--stochastic", action="store_true",
                   help="sample actions instead of taking the greedy one")
    return p.parse_args()


def main():
    args = parse_args()
    render_mode = None if args.render == "none" else "human"
    env = ValetParkEnv(render_mode=render_mode, game_time=args.game_time,
                       frame_skip=args.frame_skip, num_clients=args.num_clients,
                       max_cars=args.max_cars, arrival_prob=args.arrival_prob,
                       exit_prob=args.exit_prob)
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
