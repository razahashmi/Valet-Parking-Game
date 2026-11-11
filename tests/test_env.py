"""Headless tests for the RL environment: Gym contract, anti-cheat, reward shaping,
and that the default configurations can actually be won.

    python tests/test_env.py     # prints a summary
    pytest tests/test_env.py
"""
import contextlib
import io
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import numpy as np
from stable_baselines3.common.env_checker import check_env
import rl.valet_env as valet_env
from rl.valet_env import ValetParkEnv, NOOP, UP, DOWN, LEFT, RIGHT, SPACE
from rl.train import parse_args, env_kwargs


def _quiet():
    return contextlib.redirect_stdout(io.StringIO())  # Car prints on pickup/delivery


def test_check_env_passes():
    e = ValetParkEnv(render_mode=None, game_time=20)
    check_env(e)
    e.close()


def test_obs_size_and_parked_bit():
    e = ValetParkEnv(render_mode=None, game_time=60, num_clients=1, max_cars=10)
    e.reset(seed=0)
    assert e.observation_space.shape == (115,), e.observation_space.shape  # 5 + 11*10
    e._spawn_next()
    c = e.car.sprites()[0]
    c.ClientEntered = True
    base = valet_env.GLOBAL_OBS + e.car_slot[c] * valet_env.SLOT_SIZE
    assert e._get_obs()[base + 8] == 0.0, "parked bit set before parking"
    c.active = False
    c.rect.center = e.spot_xy[c.ParkingSpot]
    c.update(e._surface)                       # latch parked
    assert c.parked
    assert e._get_obs()[base + 8] == 1.0, "parked bit not set after parking"
    e.close()


def test_obs_target_offset_points_to_spot_then_exit():
    e = ValetParkEnv(render_mode=None, game_time=60, num_clients=1, max_cars=10)
    e.reset(seed=0)
    with _quiet():
        e._spawn_next()
    c = e.car.sprites()[0]
    c.ClientEntered = True
    base = valet_env.GLOBAL_OBS + e.car_slot[c] * valet_env.SLOT_SIZE
    sx, sy = e.spot_xy[c.ParkingSpot]
    c.rect.center = (sx - 200, sy + 100)       # 200 px left of and 100 px below its spot
    o = e._get_obs()
    assert np.isclose(o[base + 9], 200 / valet_env.WIDTH)
    assert np.isclose(o[base + 10], -100 / valet_env.HEIGHT)
    c.active = False
    c.rect.center = (sx, sy)
    c.update(e._surface)                       # parked: the target becomes the exit
    o = e._get_obs()
    # Exit is up and to the right (right-column spots may already be far enough right).
    assert o[base + 9] >= 0 and o[base + 10] < 0, "parked car should point to the exit"
    c.rect.x, c.rect.y = 1130, 125             # inside the delivery zone
    o = e._get_obs()
    assert o[base + 9] == 0.0 and o[base + 10] == 0.0
    e.close()


def test_unparked_client_is_never_called():
    """Anti-cheat: a client does not come out until their car is parked at its spot."""
    e = ValetParkEnv(render_mode=None, game_time=120, num_clients=1,
                     exit_prob=1.0, arrival_prob=0.0)
    e.reset(seed=0)
    e._spawn_next()
    c = e.car.sprites()[0]
    for _ in range(e.exit_ticks + 5):          # past a pickup event, car left unparked
        e._tick(NOOP, do_toggle=True)
    assert c.ClientEntered
    assert not c.parked
    assert not c.Client.sprite.ClientExited, "unparked car's client was called (cheatable)"
    e.close()


def test_parked_client_is_called():
    e = ValetParkEnv(render_mode=None, game_time=120, num_clients=1,
                     exit_prob=1.0, arrival_prob=0.0)
    e.reset(seed=0)
    e._spawn_next()
    c = e.car.sprites()[0]
    for _ in range(500):                        # let the client finish walking in
        e._tick(NOOP, do_toggle=True)
    assert c.ClientEntered
    c.active = False
    c.rect.center = e.spot_xy[c.ParkingSpot]     # park at the assigned spot
    called = False
    for _ in range(e.exit_ticks + 5):
        e._tick(NOOP, do_toggle=True)
        if c.Client.sprite.ClientExited:
            called = True
            break
    assert c.parked and called, "parked car's client was never called"
    e.close()


def test_pickup_skips_clients_still_walking_in():
    """A pickup must go to a client who is inside the shop, not be wasted on one who is
    still walking in (Car.ClientExit ignores those)."""
    for seed in range(10):
        e = ValetParkEnv(render_mode=None, num_clients=2, exit_prob=1.0, arrival_prob=0.0,
                         exit_interval_s=0.5)
        e.reset(seed=seed)
        with _quiet():
            e._spawn_next(); e._spawn_next()
        walking, inside = sorted(e.car.sprites(), key=lambda s: e.car_slot[s])
        for c in (walking, inside):
            c.rect.center = e.spot_xy[c.ParkingSpot]   # both parked right away
        inside.ClientEntered = True                    # this client is already in the shop
        with _quiet():
            for _ in range(e.exit_ticks):
                e._tick(NOOP, do_toggle=False)
        assert not walking.ClientEntered, "test setup: client finished walking too soon"
        assert inside.Client.sprite.ClientExited, f"seed {seed}: pickup was wasted"
        e.close()


def _farm_return(cycle, steps=225):
    e = ValetParkEnv(render_mode=None, game_time=60, num_clients=1,
                     arrival_prob=0, exit_prob=0, frame_skip=4)
    e.reset(seed=0)
    with _quiet():
        e._spawn_next()
    c = e.car.sprites()[0]
    c.ClientEntered = True
    c.rect.center = (700, 450)                  # in the lot, far from its spot
    e.player.sprite.rect.center = c.rect.center
    e.step(SPACE)                               # enter once (pays the one-off bonus)...
    e.step(SPACE)                               # ...and step out; not counted below
    total = 0.0
    for i in range(steps):
        _, r, terminated, truncated, _ = e.step(cycle[i % len(cycle)])
        total += r
        if terminated or truncated:
            break
    e.close()
    return total


def test_enter_exit_cycling_earns_nothing():
    """Regression: stepping toward a car, entering and leaving used to farm walk shaping
    (+7 vs -9 for doing nothing over 225 steps) because leaving teleports the player."""
    idle = _farm_return([NOOP])
    farm = _farm_return([UP, SPACE, SPACE])
    assert farm <= idle + 1e-6, f"cycling earns {farm - idle:+.3f} over idling"


def test_shaping_is_exact_potential_difference():
    """Total progress shaping over an episode == Phi(end) - Phi(start), for any actions."""
    saved = valet_env.WAIT_PENALTY, valet_env.ENTER_BONUS
    valet_env.WAIT_PENALTY = valet_env.ENTER_BONUS = 0.0   # isolate the potential term
    try:
        rng = np.random.default_rng(1)
        # Drift up-left (toward the entrance) with frequent SPACE so the run enters,
        # drives and leaves cars rather than only walking.
        actions = rng.choice([NOOP, UP, DOWN, LEFT, RIGHT, SPACE], size=4000,
                             p=[.05, .25, .1, .3, .1, .2])
        returns, ends = [], []
        for shaping in (True, False):
            e = ValetParkEnv(render_mode=None, game_time=40, num_clients=4,
                             arrival_interval_s=3.0, exit_interval_s=6.0,
                             reward_shaping=shaping)
            e.reset(seed=3)
            phi0, total, drove = e._potential(), 0.0, 0
            with _quiet():
                for a in actions:
                    _, r, terminated, truncated, _ = e.step(int(a))
                    total += r
                    drove += e.driving
                    if terminated or truncated:
                        break
            returns.append(total)
            ends.append(e._potential() - phi0)
            e.close()
        assert drove > 0, "test did not exercise driving"
        assert abs((returns[0] - returns[1]) - ends[0]) < 1e-6, (returns, ends)
    finally:
        valet_env.WAIT_PENALTY, valet_env.ENTER_BONUS = saved


def test_walk_guidance_ignores_parked_cars():
    """The on-foot target is the car that needs work, not merely the nearest car."""
    e = ValetParkEnv(render_mode=None, num_clients=2, arrival_prob=0, exit_prob=0)
    e.reset(seed=0)
    with _quiet():
        e._spawn_next(); e._spawn_next()
    parked, arrival = sorted(e.car.sprites(), key=lambda s: e.car_slot[s])
    for c in (parked, arrival):
        c.ClientEntered = True
    parked.rect.center = e.spot_xy[parked.ParkingSpot]
    parked.parked = True                                    # done; nobody waiting
    arrival.rect.center = valet_env.ENTRANCE_REST           # blocking the entrance
    e.player.sprite.rect.center = (parked.rect.centerx, parked.rect.centery + 120)
    assert e._focus_car()[0] is arrival
    parked.Client.sprite.ClientExited = True                # now its client is waiting
    assert e._focus_car()[0] is parked                      # nearer, and needs delivering
    e.close()


def test_timeout_terminates_as_a_loss():
    e = ValetParkEnv(render_mode=None, game_time=3, num_clients=1, frame_skip=4)
    e.reset(seed=0)
    terminated = truncated = False
    while not (terminated or truncated):
        _, _, terminated, truncated, info = e.step(NOOP)
    assert terminated and not truncated and not info["won"]
    e.close()


def _oracle_win_rate(kwargs, seeds):
    """Win rate of perfect (teleporting) play: an upper bound for any policy."""
    wins = 0
    for seed in seeds:
        e = ValetParkEnv(**kwargs)
        e.reset(seed=seed)
        done = False
        with _quiet():
            while not done:
                for c in e.car.sprites():
                    c.active = False
                    if c.parked and c.Client.sprite.ClientExited:
                        c.rect.x, c.rect.y = 1130, 125           # to the exit corner
                    elif not c.parked:
                        c.rect.center = e.spot_xy[c.ParkingSpot]  # to its spot
                _, _, terminated, truncated, info = e.step(NOOP)
                done = terminated or truncated
        wins += info["won"]
        e.close()
    return wins / len(seeds)


def test_training_configs_are_winnable():
    """Regression: `python -m rl.train` used to run 60 s episodes for 10 clients with
    coin-flip pickups, which even perfect play could never win."""
    cases = [
        ([], 0.8),                                              # rl.train defaults
        (["--num-clients", "3"], 0.9),
        (["--num-clients", "1", "--arrival-prob", "1"], 1.0),   # README curriculum stage 1
        (["--num-clients", "3", "--arrival-prob", "1"], 1.0),   # README curriculum stage 2
    ]
    for argv, floor in cases:
        rate = _oracle_win_rate(env_kwargs(parse_args(argv)), seeds=range(10))
        assert rate >= floor, f"rl.train {' '.join(argv) or '(defaults)'}: oracle wins {rate:.0%}"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
