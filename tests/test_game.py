"""Headless regression tests for ValetGame.

Drives the real per-frame game logic (ValetGame.step) with scripted events over
many iterations. Run from anywhere:

    python tests/test_game.py        # prints a summary
    pytest tests/test_game.py        # or via pytest
"""
import os
import sys
import random

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)                       # game assets load from the repo root
sys.path.insert(0, ROOT)

import pygame
from ValetParkGame import ValetGame

DT = 1 / 60.0
KD, KU = pygame.KEYDOWN, pygame.KEYUP


def _ev(t, **kw):
    return pygame.event.Event(t, **kw)


# Parking-spot pixel centres (from ParkingSpots.DrawParkingSpots geometry).
SPOT_XY = {}
for _i in range(6):
    SPOT_XY[1001 + _i] = (430 + _i * 100, 375)
    SPOT_XY[1007 + _i] = (430 + _i * 100, 525)
for _i in range(3):
    SPOT_XY[1013 + _i] = (167, 350 + _i * 100)
    SPOT_XY[1016 + _i] = (1202, 350 + _i * 100)


def test_flag_reset_on_reentry():
    """Re-entering a car must not make it drive by itself (stale drive-flag bug)."""
    g = ValetGame(present=False, num_clients=1, game_time=60)
    g.spawn_next_car()
    c = g.car.sprites()[0]
    c.ClientEntered = True
    g.player.sprite.rect.center = c.rect.center
    g.handle_event(_ev(KD, key=pygame.K_SPACE))          # enter
    assert g.Car_select and c.active
    g.handle_event(_ev(KD, key=pygame.K_UP))             # hold forward
    g.step([], DT)                                       # drives
    g.handle_event(_ev(KD, key=pygame.K_SPACE))          # exit while 'holding' UP
    assert c.activeforward is False and c.direction == 0
    g.player.sprite.rect.center = c.rect.center
    g.handle_event(_ev(KD, key=pygame.K_SPACE))          # re-enter
    x = c.rect.centerx
    g.step([], DT)
    assert c.rect.centerx == x, "car drove by itself after re-entering"


def test_space_without_car_does_not_crash():
    g = ValetGame(present=False, num_clients=3, game_time=60)
    g.handle_event(_ev(KD, key=pygame.K_SPACE))
    g.step([], DT)


def test_pickup_reaches_every_client():
    """Each pickup event calls a not-yet-called client, so all are eventually served."""
    n, events = 10, 15                       # 15 pickup events over a 300s game
    for seed in range(50):
        rng = random.Random(seed)
        g = ValetGame(present=False, num_clients=n, game_time=300)
        for i in range(n):                   # all cars present and entered
            g.spawn_next_car()
        for c in g.car.sprites():
            c.ClientEntered = True
        for _ in range(events):
            g.handle_event(_ev(g.Car_exit))
        called = sum(c.Client.sprite.ClientExited for c in g.car.sprites())
        assert called == n, f"only {called}/{n} clients called"


class _FakeKeys:
    def __init__(self):
        self.down = set()

    def __getitem__(self, k):
        return k in self.down


def test_random_input_never_crashes():
    """Fuzz: random keys/events for a full game; must reach a terminal state, no crash."""
    for seed in range(4):
        random.seed(seed)
        g = ValetGame(present=False, num_clients=5, game_time=30)
        keys = _FakeKeys()
        pygame.key.get_pressed = lambda: keys
        frame, max_frames = 0, int(30 / DT) + 300
        while g.game_state == "playing" and frame < max_frames:
            events = []
            if frame % 600 == 0:
                events.append(_ev(g.Car_enter))
            if frame % 1200 == 0:
                events.append(_ev(g.Car_exit))
            if random.random() < 0.05:
                events.append(_ev(KD, key=pygame.K_SPACE))
            for k in (pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT):
                if random.random() < 0.1:
                    events.append(_ev(KD, key=k)); keys.down.add(k)
                if random.random() < 0.1:
                    events.append(_ev(KU, key=k)); keys.down.discard(k)
            g.step(events, DT)
            frame += 1
        assert g.game_state in ("won", "lost")


def _auto_play(seed, num_clients, game_time):
    """Perfect teleport play: park arrived cars, deliver called ones."""
    random.seed(seed)
    g = ValetGame(present=False, num_clients=num_clients, game_time=game_time)
    g.player.sprite.active = False
    frame, max_frames = 0, int(game_time / DT) + 300
    while g.game_state == "playing" and frame < max_frames:
        events = []
        if frame > 0 and frame % 600 == 0:
            events.append(_ev(g.Car_enter))
        if frame > 0 and frame % 1200 == 0:
            events.append(_ev(g.Car_exit))
        for c in g.car.sprites():
            if not c.ClientEntered:
                continue
            if c.Client.sprite.ClientExited and not c.SuccessDelivery:
                c.active = False
                c.rect.x, c.rect.y = 1130, 125           # deliver at the exit corner
            elif not c.Client.sprite.ClientExited:
                c.active = False
                c.rect.center = SPOT_XY[c.ParkingSpot]    # park off the entrance
        g.step(events, DT)
        frame += 1
    return g.game_state


def test_game_is_winnable_with_perfect_play():
    # Arrivals are 50% random, so a rare seed can run late; require a clear majority
    # (a broken pickup mechanic would make this 0). The default 10-client/300s game
    # wins on every seed tried.
    seeds = range(6)
    wins = sum(_auto_play(s, num_clients=6, game_time=300) == "won" for s in seeds)
    assert wins >= 5, f"only {wins}/{len(seeds)} winnable with perfect play"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
