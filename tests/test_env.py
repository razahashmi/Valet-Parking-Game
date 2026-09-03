"""Headless tests for the RL environment, focused on the Gym contract and anti-cheat.

    python tests/test_env.py     # prints a summary
    pytest tests/test_env.py
"""
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from stable_baselines3.common.env_checker import check_env
from rl.valet_env import ValetParkEnv, NOOP


def test_check_env_passes():
    e = ValetParkEnv(render_mode=None, game_time=20)
    check_env(e)
    e.close()


def test_obs_size_and_parked_bit():
    e = ValetParkEnv(render_mode=None, game_time=60, num_clients=1, max_cars=10)
    e.reset(seed=0)
    assert e.observation_space.shape == (95,), e.observation_space.shape  # 5 + 9*10
    e._spawn_next()
    c = e.car.sprites()[0]
    c.ClientEntered = True
    base = 5 + e.car_slot[c] * 9
    assert e._get_obs()[base + 8] == 0.0, "parked bit set before parking"
    c.active = False
    c.rect.center = e.spot_xy[c.ParkingSpot]
    c.update(e._surface)                       # latch parked
    assert c.parked
    assert e._get_obs()[base + 8] == 1.0, "parked bit not set after parking"
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


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
