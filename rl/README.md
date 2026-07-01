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

**Observation** — `Box(-1, 1, (5 + 8·max_cars,))`, float32:
`[player_x, player_y, driving, time_remaining, entrance_blocked]` then one **fixed slot
per client** (slot = client index, stable across the whole episode even as cars are
delivered), each `[present, car_x, car_y, sin(angle), cos(angle), parking_spot,
client_waiting, is_active]`. Positions are normalized by the screen size; empty slots are
zero. `max_cars` fixes the vector length so one policy transfers across a curriculum.

**Reward** (per step, tunable at the top of `valet_env.py`):
`-0.01` step cost · `+10` per delivery · `-1` per collision · `-0.05` while the entrance
is blocked · `+50` win bonus. With `reward_shaping=True` (default) dense guidance is added
across the whole task: potential-based progress toward the current target (walk to the car
that needs attention → drive it to its spot → drive to the exit once the client is waiting),
a one-off `+1` for first entering each car, and `-0.02`/tick per waiting client. The episode
`terminates` when every client has been served and `truncates` when time runs out.

## Constructor options

`ValetParkEnv(render_mode=None, game_time=300, fps=60, arrival_interval_s=10,
exit_interval_s=20, num_clients=10, num_car_models=12, reward_shaping=True,
frame_skip=1, max_cars=None, arrival_prob=0.5, exit_prob=0.5)`

- `frame_skip` — hold each action for N ticks (shortens the horizon; training uses 4).
- `num_clients` — clients per episode; lower it for a curriculum.
- `max_cars` — observation capacity (defaults to `num_clients`); keep it fixed across a
  curriculum so one policy keeps working as `num_clients` grows.
- `arrival_prob` / `exit_prob` — chance a scheduled arrival / pickup fires. At the default
  `0.5`, an episode can be unwinnable by luck (the client never requests pickup in time);
  set both to `1.0` for deterministic timing so a competent policy can approach 100%.

Lower `game_time` / `arrival_interval_s` for faster iteration; set `reward_shaping=False`
to train against the raw sparse reward.

## Known limitations (good next improvements)

- **Sim and draw are coupled.** `Car.update()` both moves cars and blits the client
  animation, so headless steps do a few wasted blits (throughput is still ~75k/s).
- **Rect-based collisions.** Uses shrunk bounding boxes (`collide_rect_ratio(0.7)`),
  approximate for rotated cars; mask-based would be more precise.
- **The spot is handed to the agent.** The observation includes each car's parking spot,
  so there is nothing to *remember*. To study the memorization question in the top-level
  README, add a mode that hides the spot and exposes car-model / client-avatar ids instead.

## Training

PPO via [Stable-Baselines3](https://stable-baselines3.readthedocs.io/). Run everything
from the repo root:

```bash
python -m rl.train                                    # 1M steps, 4 envs, frame_skip=4
python -m rl.train --timesteps 5_000_000 --n-envs 8   # longer run
# curriculum: learn 1 client with deterministic timing + LR decay, then resume with more
python -m rl.train --num-clients 1 --arrival-prob 1 --exit-prob 1 --lr-schedule --logdir rl/runs/s1
python -m rl.train --num-clients 3 --load rl/runs/s1/valet_ppo_final --logdir rl/runs/s3
tensorboard --logdir rl/runs                          # watch curves
```

Outputs land in `rl/runs/` (gitignored): periodic `checkpoints/`, the best-by-eval
`best/best_model.zip`, a `valet_ppo_final.zip`, and tensorboard logs.

Watch or evaluate a saved agent (match `--frame-skip` / `--max-cars` to training):

```bash
python -m rl.enjoy rl/runs/best/best_model.zip              # render in a window
python -m rl.enjoy rl/runs/best/best_model.zip --render none --episodes 20  # stats only
```

Tips: start with a short `--game-time` (30–60s) and a small `--num-clients`, then scale up.
Judge progress by `delivered`/`win` in `enjoy`, **not** raw reward (shaping inflates it).
Headless throughput is ~75k steps/s, so the network — not the sim — is the bottleneck;
`--n-envs` mainly buys batch diversity, and `--subproc` gives true process parallelism.
