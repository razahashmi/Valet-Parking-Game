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

**Observation** — `Box(-1, 1, (5 + 11·max_cars,))`, float32:
`[player_x, player_y, driving, time_remaining, entrance_blocked]` then one **fixed slot
per client** (slot = client index, stable across the whole episode even as cars are
delivered), each `[present, car_x, car_y, sin(angle), cos(angle), parking_spot,
client_waiting, is_active, parked, target_dx, target_dy]`. `target_dx/dy` is how far the
car still has to move to reach its current target: its spot until it is parked, then the
exit zone (zero once inside). Positions and offsets are normalized by the screen size; empty
slots are zero. `max_cars` fixes the vector length so one policy transfers across a curriculum.

**Task is enforced (no cheating):** a car must actually be parked at its assigned numbered
spot before its client is called and before it can be delivered — the agent cannot win by
shuttling cars straight from the entrance to the exit. `parked` reports this per car.

**Reward** (per tick, tunable at the top of `valet_env.py`):
`-0.01` step cost · `+10` per delivery · `-1` per collision · `-0.05` while the entrance
is blocked · `+50` win bonus. With `reward_shaping=True` (default) dense guidance is added:

- **Progress (potential-based).** Each tick adds `Φ(s′) − Φ(s)`, where `Φ` is computed
  from the current state only: `0.02 ×` the pixels of work done (each car's path from the
  entrance to its spot and on to the exit; once parked, its distance to the exit) minus
  `0.01 ×` the distance from the valet (the player, or the car being driven) to the nearest
  *other* car that still needs work (not yet parked, or its client is waiting). Because `Φ`
  is a function of state, any loop (e.g. stepping out of a car and back in) earns exactly
  zero, and an episode's total progress shaping equals the progress actually made.
- A one-off `+1` the first time each car is entered, and `-0.02`/tick per waiting client.

The episode **terminates** when every client has been served (a win) or when the clock runs
out (a loss); `info["won"]` tells them apart. Running out of time is a real game-over (the
remaining time is in the observation), so it is never reported as `truncated`.

## Constructor options

`ValetParkEnv(render_mode=None, game_time=None, fps=60, arrival_interval_s=10,
exit_interval_s=20, num_clients=10, num_car_models=12, reward_shaping=True,
frame_skip=1, max_cars=None, arrival_prob=0.5, exit_prob=1.0)`

- `game_time` — seconds per episode. `None` (default) scales it with the number of clients,
  `60 + 24 × num_clients` (1 → 84 s, 3 → 132 s, 10 → 300 s, the human game's length), so a
  perfect player can serve everyone; see `default_game_time`.
- `frame_skip` — hold each action for N ticks (shortens the horizon; training uses 4).
- `num_clients` — clients per episode; lower it for a curriculum.
- `max_cars` — observation capacity (defaults to `num_clients`); keep it fixed across a
  curriculum so one policy keeps working as `num_clients` grows.
- `arrival_prob` / `exit_prob` — chance a scheduled arrival / pickup fires. The defaults
  match the human game: arrivals are a coin flip every 10 s, and every pickup event (every
  20 s) calls one parked client. With random arrivals a rare unlucky episode still can't be
  finished in time (a perfect-play oracle wins ~90% at 10 clients, ≥95% at 1–5); set
  `arrival_prob=1.0` for deterministic timing, where the oracle wins 100%. Lowering
  `exit_prob` below 1 makes many episodes unwinnable.

Lower `game_time` / `arrival_interval_s` for faster iteration; set `reward_shaping=False`
to train against the raw sparse reward.

## Known limitations (good next improvements)

- **Sim and draw are coupled.** `Car.update()` both moves cars and blits the client
  animation, so headless steps do a few wasted blits (throughput is still ~75k/s).
- **Rect-based collisions.** Uses shrunk bounding boxes (`collide_rect_ratio(0.7)`),
  approximate for rotated cars; mask-based would be more precise.
- **The spot is handed to the agent (memory not yet required).** Parking at the correct
  spot is enforced, but the spot number *and* the offset to it are in the observation every
  frame, so the agent never has to *remember* anything. (The offset is what makes the task
  learnable: with only the spot ID, PPO did not learn curriculum stage 1 in 1.5M steps.) To
  study the memorization question in the top-level README, add a mode that shows the spot
  only while the client's speech bubble is visible, announces pickups by client-avatar id
  rather than on the car's own slot, exposes car-model ids, and uses a recurrent policy.

## Training

PPO via [Stable-Baselines3](https://stable-baselines3.readthedocs.io/). Run everything
from the repo root:

```bash
python -m rl.train                                    # 1M steps, 4 envs, frame_skip=4
python -m rl.train --timesteps 5_000_000 --n-envs 8   # longer run
# curriculum: learn 1 client with deterministic timing + LR decay, then resume with more
python -m rl.train --num-clients 1 --arrival-prob 1 --lr-schedule --logdir rl/runs/s1
python -m rl.train --num-clients 3 --arrival-prob 1 --load rl/runs/s1/valet_ppo_final --logdir rl/runs/s3
tensorboard --logdir rl/runs                          # watch curves
```

Episode length follows `--num-clients` unless you pass `--game-time` (see `game_time`
above). Outputs land in `rl/runs/` (gitignored): periodic `checkpoints/`, the
`best/best_model.zip` with the highest eval **win rate** (then deliveries; shaped reward
only breaks ties), a `valet_ppo_final.zip`, and tensorboard logs (`eval/win_rate`,
`eval/delivered_mean`, `rollout/win_rate`, `rollout/delivered_mean`).

Evaluation and `enjoy` **sample** actions from the policy, as in training. Greedy (argmax)
play can lock up in this env — e.g. standing just short of a car pressing SPACE, which
changes nothing, so the greedy action never changes either; one stage-1 agent won 95% of
episodes sampled and 0% greedy. Pass `--deterministic` to `enjoy` to see greedy play.

Watch or evaluate a saved agent (`enjoy` takes the same environment flags as `train`; pass
the ones you trained with):

```bash
python -m rl.enjoy rl/runs/best/best_model.zip              # render in a window
python -m rl.enjoy rl/runs/s1/best/best_model.zip --num-clients 1 --arrival-prob 1 --render none --episodes 20
```

Tips: start with a small `--num-clients` (the episode length scales with it), then scale up.
Judge progress by `delivered`/`win` in `enjoy`, **not** raw reward (shaping inflates it).
Headless throughput is ~75k steps/s, so the network — not the sim — is the bottleneck;
`--n-envs` mainly buys batch diversity, and `--subproc` gives true process parallelism.
