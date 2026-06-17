"""Train a PPO agent on ValetParkEnv with Stable-Baselines3.

Examples (run from the repo root):
    python -m rl.train                                  # 1M steps, sensible defaults
    python -m rl.train --timesteps 5_000_000 --n-envs 8
    python -m rl.train --load rl/runs/best/best_model   # resume / fine-tune

Watch progress with:   tensorboard --logdir rl/runs
"""
import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from rl.valet_env import ValetParkEnv


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--game-time", type=int, default=60, help="seconds per episode")
    p.add_argument("--subproc", action="store_true",
                   help="use SubprocVecEnv (one process per env) instead of DummyVecEnv")
    p.add_argument("--logdir", default="rl/runs")
    p.add_argument("--load", default=None, help="path to a saved .zip to resume from")
    p.add_argument("--seed", type=int, default=0)
    # PPO hyper-parameters (defaults are fine for a first run)
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.999, help="high: episodes are long")
    p.add_argument("--ent-coef", type=float, default=0.01, help="exploration bonus")
    p.add_argument("--progress", action="store_true", help="show a tqdm progress bar")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.logdir, exist_ok=True)
    env_kwargs = dict(render_mode=None, game_time=args.game_time)
    vec_cls = SubprocVecEnv if args.subproc else DummyVecEnv

    env = make_vec_env(ValetParkEnv, n_envs=args.n_envs, seed=args.seed,
                       env_kwargs=env_kwargs, vec_env_cls=vec_cls)
    eval_env = make_vec_env(ValetParkEnv, n_envs=1, seed=args.seed + 1000,
                            env_kwargs=env_kwargs, vec_env_cls=DummyVecEnv)

    callbacks = [
        CheckpointCallback(save_freq=max(100_000 // args.n_envs, 1),
                           save_path=os.path.join(args.logdir, "checkpoints"),
                           name_prefix="valet_ppo"),
        EvalCallback(eval_env,
                     best_model_save_path=os.path.join(args.logdir, "best"),
                     log_path=os.path.join(args.logdir, "eval"),
                     eval_freq=max(50_000 // args.n_envs, 1),
                     n_eval_episodes=5, deterministic=True),
    ]

    if args.load:
        print(f"resuming from {args.load}")
        model = PPO.load(args.load, env=env, tensorboard_log=args.logdir)
    else:
        model = PPO(
            "MlpPolicy", env, verbose=1, seed=args.seed,
            tensorboard_log=args.logdir,
            n_steps=args.n_steps, batch_size=args.batch_size,
            learning_rate=args.lr, gamma=args.gamma, gae_lambda=0.95,
            ent_coef=args.ent_coef, n_epochs=10,
            policy_kwargs=dict(net_arch=[256, 256]),
        )

    model.learn(total_timesteps=args.timesteps, callback=callbacks,
                progress_bar=args.progress)

    final_path = os.path.join(args.logdir, "valet_ppo_final")
    model.save(final_path)
    env.close()
    eval_env.close()
    print(f"\nDone. Final model: {final_path}.zip")
    print(f"Best model:  {os.path.join(args.logdir, 'best', 'best_model.zip')}")
    print(f"Watch it:    python -m rl.enjoy {final_path}.zip")


if __name__ == "__main__":
    main()
