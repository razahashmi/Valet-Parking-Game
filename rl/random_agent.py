"""Smoke test: drive ValetParkEnv with random actions and sanity-check the Gym API.

Run from the repo root:   python -m rl.random_agent
"""
import numpy as np

from rl.valet_env import ValetParkEnv


def run_episode(env, seed=None):
    obs, info = env.reset(seed=seed)
    assert env.observation_space.contains(obs), "initial observation is out of bounds"
    total_reward, steps = 0.0, 0
    terminated = truncated = False
    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs), f"observation out of bounds at step {steps}"
        total_reward += reward
        steps += 1
    return steps, total_reward, info


def rollout_observations(env, seed, actions):
    obs, _ = env.reset(seed=seed)
    history = [obs]
    for a in actions:
        obs, _, terminated, truncated, _ = env.step(a)
        history.append(obs)
        if terminated or truncated:
            break
    return history


def main():
    # Short episodes with brisk arrivals keep the smoke test fast while still
    # exercising spawning, the entrance-blocked path, and car-bearing observations.
    env = ValetParkEnv(render_mode=None, game_time=15,
                       arrival_interval_s=3.0, exit_interval_s=6.0)
    print(f"observation_space: {env.observation_space}")
    print(f"action_space:      {env.action_space}")

    # Determinism: same seed + same actions -> identical trajectories.
    rng = np.random.default_rng(0)
    fixed_actions = [int(rng.integers(0, 6)) for _ in range(700)]
    a = rollout_observations(env, 123, fixed_actions)
    b = rollout_observations(env, 123, fixed_actions)
    assert len(a) == len(b) and all(np.allclose(x, y) for x, y in zip(a, b)), \
        "env is not deterministic for a fixed seed + action sequence"
    print(f"determinism over {len(a)} steps: OK")

    for ep in range(3):
        steps, total_reward, info = run_episode(env, seed=ep)
        print(f"episode {ep}: steps={steps:5d}  return={total_reward:8.2f}  "
              f"delivered={info['delivered_total']}/{env.total_clients}  "
              f"entered={info['cars_entered']}")
    env.close()

    # rgb_array rendering returns a proper (H, W, 3) frame.
    renv = ValetParkEnv(render_mode="rgb_array", game_time=2)
    renv.reset(seed=0)
    renv.step(renv.action_space.sample())
    frame = renv.render()
    assert frame.shape == (768, 1366, 3), f"unexpected frame shape {frame.shape}"
    print(f"rgb_array render: OK {frame.shape}")
    renv.close()

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
