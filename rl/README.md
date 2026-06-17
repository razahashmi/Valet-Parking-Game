# Valet-Park RL environment

A [Gymnasium](https://gymnasium.farama.org/) environment that wraps the Valet-Park
game so a reinforcement-learning agent can play it. It reuses the game's own sprite
classes (`Car`, `Player`, `Spots`) from `src/` and replaces the real-time event loop
with a deterministic, step-based simulation.

## Quick start

```bash
pip install -r requirements.txt
python -m rl.random_agent      # smoke test: random agent + API checks
```

```python
from rl.valet_env import ValetParkEnv

env = ValetParkEnv(render_mode=None)        # headless (fast); use "human" to watch
obs, info = env.reset(seed=0)
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
env.close()
```

It is also registered with Gymnasium: `import rl; gym.make("ValetPark-v0")`.

## Spaces

**Action** — `Discrete(6)`, mirroring the keyboard controls:

| id | walking          | driving a car      |
|----|------------------|--------------------|
| 0  | no-op            | coast              |
| 1  | walk up          | drive forward      |
| 2  | walk down        | drive backward     |
| 3  | walk left        | steer left         |
| 4  | walk right       | steer right        |
| 5  | enter the car you're standing on | step out of the car |

**Observation** — `Box(-1, 1, (5 + 8·num_clients,))`, float32:
`[player_x, player_y, driving, time_remaining, entrance_blocked]` then, per car,
`[present, car_x, car_y, sin(angle), cos(angle), parking_spot, client_waiting, is_active]`.
Positions are normalized by the screen size; absent car slots are zero.

**Reward** (per step, tunable at the top of `valet_env.py`):
`-0.01` step cost · `+10` per delivery · `-1` per collision the driven car causes ·
`-0.05` while the entrance is blocked · `+50` win bonus. With `reward_shaping=True`
(default) a small potential-based term also rewards the driven car for getting closer
to its goal (its parking spot, then the exit once the client comes out) — this makes
the long delivery sequence learnable. The episode `terminates` when every client has
been served and `truncates` when time runs out.

## Constructor options

`ValetParkEnv(render_mode=None, game_time=300, fps=60, arrival_interval_s=10,
exit_interval_s=20, num_clients=10, num_car_models=12, reward_shaping=True)`

Lower `game_time` / `arrival_interval_s` for faster iteration during development.
Set `reward_shaping=False` to train against the raw sparse reward.

## Known limitations (good first improvements)

- **Sim and draw are coupled.** `Car.update()` both moves cars and blits the client
  animation, so headless steps still do a few wasted blits. Splitting render out of
  `Car` would speed up training meaningfully.
- **Rect-based collisions.** Uses shrunk bounding boxes (`collide_rect_ratio(0.7)`),
  which is approximate for rotated cars. Mask-based collision would be more precise.
- **One window per process.** pygame allows a single display, so vectorized training
  needs subprocess envs (e.g. SB3 `SubprocVecEnv`).
- **Sparse reward.** Delivering a car requires a long, precise sequence; consider
  adding shaping (distance-to-spot, distance-to-exit, pickup bonus) to bootstrap learning.

## Training

PPO via [Stable-Baselines3](https://stable-baselines3.readthedocs.io/). Run everything
from the repo root:

```bash
python -m rl.train                                  # 1M steps, 4 envs, game_time=60
python -m rl.train --timesteps 5_000_000 --n-envs 8 # longer run
python -m rl.train --load rl/runs/best/best_model   # resume / fine-tune
tensorboard --logdir rl/runs                        # watch curves
```

Outputs land in `rl/runs/` (gitignored): periodic `checkpoints/`, the best-by-eval
`best/best_model.zip`, a `valet_ppo_final.zip`, and tensorboard logs.

Watch or evaluate a saved agent:

```bash
python -m rl.enjoy rl/runs/best/best_model.zip              # render in a window
python -m rl.enjoy rl/runs/best/best_model.zip --render none --episodes 20  # stats only
```

Tips: start with a short `--game-time` (e.g. 30–60s) and `reward_shaping=True` (default).
Headless env throughput is ~75k steps/s, so the neural network — not the simulation — is
the bottleneck; `--n-envs` mainly buys batch diversity. For true process-parallel envs use
`--subproc`.
