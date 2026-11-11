"""Train a PPO agent on ValetParkEnv with Stable-Baselines3.

Examples (run from the repo root):
    python -m rl.train                                  # 1M steps, sensible defaults
    python -m rl.train --timesteps 5_000_000 --n-envs 8
    python -m rl.train --load rl/runs/best/best_model   # resume / fine-tune

Watch progress with:   tensorboard --logdir rl/runs
"""
import argparse
import os

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import safe_mean

from rl.valet_env import ValetParkEnv


class RolloutStats(BaseCallback):
    """Log the true task metrics (deliveries, win rate) to TensorBoard each rollout.

    Raw reward is inflated by shaping, so judge progress by these instead.
    """

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        buf = self.model.ep_info_buffer
        if not buf:
            return
        delivered = [e["delivered_total"] for e in buf if "delivered_total" in e]
        won = [float(e["won"]) for e in buf if "won" in e]
        if delivered:
            self.logger.record("rollout/delivered_mean", safe_mean(delivered))
        if won:
            self.logger.record("rollout/win_rate", safe_mean(won))


class TaskEvalCallback(BaseCallback):
    """Periodically evaluate the policy and keep the best model by *task success*.

    SB3's EvalCallback ranks models by mean (shaped) reward, which can prefer a policy
    that games the shaping over one that delivers cars. This ranks by win rate, then
    mean deliveries, with mean reward only as the last tie-breaker.

    Actions are *sampled* by default, as in training. A greedy (argmax) policy can lock
    up here: e.g. standing just short of a car pressing SPACE, which changes nothing, so
    the argmax never changes either. One stage-1 agent won 95% sampled and 0% greedy.
    """

    def __init__(self, eval_env, eval_freq, n_eval_episodes, best_model_save_path,
                 deterministic=False, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_model_save_path = best_model_save_path
        self.deterministic = deterministic
        self.best_key = None

    def _on_step(self):
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
            return True
        won, delivered = [], []

        def grab(locals_, _globals):
            if locals_["done"]:
                info = locals_["info"]
                won.append(float(info["won"]))
                delivered.append(info["delivered_total"])

        rewards, _ = evaluate_policy(self.model, self.eval_env,
                                     n_eval_episodes=self.n_eval_episodes,
                                     deterministic=self.deterministic,
                                     return_episode_rewards=True, callback=grab)
        key = (float(np.mean(won)), float(np.mean(delivered)), float(np.mean(rewards)))
        self.logger.record("eval/win_rate", key[0])
        self.logger.record("eval/delivered_mean", key[1])
        self.logger.record("eval/mean_reward", key[2])
        self.logger.dump(self.num_timesteps)
        if self.best_key is None or key > self.best_key:
            self.best_key = key
            os.makedirs(self.best_model_save_path, exist_ok=True)
            self.model.save(os.path.join(self.best_model_save_path, "best_model"))
            if self.verbose:
                print(f"new best model: win_rate={key[0]:.2f} delivered={key[1]:.2f} "
                      f"reward={key[2]:.1f}")
        return True


def add_env_args(p):
    """Environment flags shared by train and enjoy, so both build the same env."""
    p.add_argument("--game-time", type=float, default=None,
                   help="seconds per episode (default: scaled to --num-clients so the "
                        "episode is winnable, see valet_env.default_game_time)")
    p.add_argument("--frame-skip", type=int, default=4, help="action-repeat ticks per step")
    p.add_argument("--num-clients", type=int, default=10,
                   help="clients per episode; start low (e.g. 1) for a curriculum")
    p.add_argument("--max-cars", type=int, default=10,
                   help="observation capacity; keep fixed across curriculum stages")
    p.add_argument("--arrival-prob", type=float, default=0.5,
                   help="chance a scheduled arrival fires (0.5 = human game; "
                        "1.0 = deterministic timing)")
    p.add_argument("--exit-prob", type=float, default=1.0,
                   help="chance a scheduled pickup fires (1.0 = human game; lower values "
                        "can make episodes unwinnable)")


def env_kwargs(args, render_mode=None):
    return dict(render_mode=render_mode, game_time=args.game_time,
                frame_skip=args.frame_skip, num_clients=args.num_clients,
                max_cars=args.max_cars, arrival_prob=args.arrival_prob,
                exit_prob=args.exit_prob)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--n-envs", type=int, default=4)
    add_env_args(p)
    p.add_argument("--subproc", action="store_true",
                   help="use SubprocVecEnv (one process per env) instead of DummyVecEnv")
    p.add_argument("--logdir", default="rl/runs")
    p.add_argument("--load", default=None, help="path to a saved .zip to resume from")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-episodes", type=int, default=10,
                   help="episodes per periodic evaluation (actions sampled, as in training)")
    # PPO hyper-parameters (defaults are fine for a first run)
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-schedule", action="store_true",
                   help="linearly decay the learning rate to 0 (stabilizes late training)")
    p.add_argument("--gamma", type=float, default=0.999, help="high: episodes are long")
    p.add_argument("--ent-coef", type=float, default=0.01, help="exploration bonus")
    p.add_argument("--progress", action="store_true", help="show a tqdm progress bar")
    return p.parse_args(argv)


def linear_schedule(initial):
    """SB3 schedule: lr goes from `initial` at start to 0 at the end of training."""
    return lambda progress_remaining: progress_remaining * initial


def main():
    args = parse_args()
    os.makedirs(args.logdir, exist_ok=True)
    kwargs = env_kwargs(args)
    vec_cls = SubprocVecEnv if args.subproc else DummyVecEnv

    # info_keywords surfaces the true metrics in the episode buffer for RolloutStats.
    monitor_kwargs = dict(info_keywords=("delivered_total", "won"))
    env = make_vec_env(ValetParkEnv, n_envs=args.n_envs, seed=args.seed,
                       env_kwargs=kwargs, vec_env_cls=vec_cls,
                       monitor_kwargs=monitor_kwargs)
    eval_env = make_vec_env(ValetParkEnv, n_envs=1, seed=args.seed + 1000,
                            env_kwargs=kwargs, vec_env_cls=DummyVecEnv)
    game_time = eval_env.get_attr("game_time")[0]
    print(f"env: {args.num_clients} clients, {game_time:.0f}s episodes, "
          f"arrival_prob={args.arrival_prob}, exit_prob={args.exit_prob}, "
          f"frame_skip={args.frame_skip}")

    callbacks = [
        RolloutStats(),
        CheckpointCallback(save_freq=max(100_000 // args.n_envs, 1),
                           save_path=os.path.join(args.logdir, "checkpoints"),
                           name_prefix="valet_ppo"),
        TaskEvalCallback(eval_env,
                         eval_freq=max(50_000 // args.n_envs, 1),
                         n_eval_episodes=args.eval_episodes,
                         best_model_save_path=os.path.join(args.logdir, "best")),
    ]

    if args.load:
        print(f"resuming from {args.load}")
        model = PPO.load(args.load, env=env, tensorboard_log=args.logdir)
    else:
        lr = linear_schedule(args.lr) if args.lr_schedule else args.lr
        model = PPO(
            "MlpPolicy", env, verbose=1, seed=args.seed,
            tensorboard_log=args.logdir,
            n_steps=args.n_steps, batch_size=args.batch_size,
            learning_rate=lr, gamma=args.gamma, gae_lambda=0.95,
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
