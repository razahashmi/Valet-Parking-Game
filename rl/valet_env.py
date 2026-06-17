"""Gymnasium environment that wraps the Valet-Park game so an RL agent can play it.

The environment reuses the game's own sprite classes (Car, Player, Spots) and helper
functions from ``src/`` and replaces the wall-clock event loop with a deterministic,
step-based simulation. One ``step()`` advances the game by a single tick (frame).

Action space (Discrete(6)) mirrors the keyboard controls:
    0 no-op
    1 up      -> walk up      / drive forward (when in a car)
    2 down    -> walk down    / drive backward
    3 left    -> walk left    / steer left
    4 right   -> walk right   / steer right
    5 space   -> enter the car you are standing on / step out of the car

Observation (Box, normalized to [-1, 1]):
    [player_x, player_y, driving, time_remaining, entrance_blocked]
    followed by, for each of up to ``num_clients`` cars:
    [present, car_x, car_y, sin(angle), cos(angle), spot, client_waiting, is_active]

Run the smoke test with:   python -m rl.random_agent
"""

import os
import sys
import math
import random

# Make ``src`` importable and run from the repo root no matter where we are invoked.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pygame
import pygame.surfarray  # noqa: F401  (needed for rgb_array rendering)
import gymnasium as gym
from gymnasium import spaces

from src.Car import Car
from src.Player import Player
from src.utils import Spots, CarSelection, GameTimer
from src.ParkingSpots import DrawParkingSpots
from src.config import GameTime, Number_Clients, number_cars_available, ParkingSpots

# Actions
NOOP, UP, DOWN, LEFT, RIGHT, SPACE = range(6)

# Screen / world constants (must match the game's geometry)
WIDTH, HEIGHT = 1366, 768
PLAYER_SPEED = 3
PENALTY_RATE = 0.2          # blocked time elapses 1.2x as fast

# Reward shaping (tune these for your experiments)
STEP_COST = -0.01           # mild time pressure every tick
DELIVERY_REWARD = 10.0      # per car handed back to its client
COLLISION_PENALTY = -1.0    # per collision the driven car causes
BLOCKED_PENALTY = -0.05     # per tick the entrance is obstructed
WIN_BONUS = 50.0            # all clients served before time ran out

# Potential-based progress shaping (disable with reward_shaping=False). Rewards the
# driven car for getting closer to its goal: the assigned spot until the client comes
# out, then the exit corner.
SHAPE_K = 0.02
EXIT_TARGET = (1230, 150)


def parking_spot_centers():
    """Pixel centres of the 18 numbered spots, derived from ParkingSpots.DrawParkingSpots."""
    centers = {}
    for i in range(6):
        centers[1001 + i] = (430 + i * 100, 375)   # top row
        centers[1007 + i] = (430 + i * 100, 525)   # bottom row
    for i in range(3):
        centers[1013 + i] = (167, 350 + i * 100)    # left column
        centers[1016 + i] = (1202, 350 + i * 100)   # right column
    return centers


class ValetParkEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, game_time=GameTime, fps=60,
                 arrival_interval_s=10.0, exit_interval_s=20.0,
                 num_clients=Number_Clients, num_car_models=number_cars_available,
                 reward_shaping=True):
        super().__init__()
        assert render_mode in (None, "human", "rgb_array")
        self.render_mode = render_mode
        self.game_time = float(game_time)
        self.fps = int(fps)
        self.arrival_ticks = max(1, int(arrival_interval_s * self.fps))
        self.exit_ticks = max(1, int(exit_interval_s * self.fps))
        self.num_clients = int(num_clients)
        self.num_car_models = int(num_car_models)
        self.max_cars = self.num_clients
        self.max_ticks = int(self.game_time * self.fps)
        self.reward_shaping = bool(reward_shaping)
        self.spot_xy = parking_spot_centers()

        # The game loads assets with paths relative to the repo root.
        if not os.path.isdir("Resources"):
            os.chdir(ROOT)

        # A video mode must exist for convert_alpha(); the dummy driver keeps it off-screen.
        if render_mode != "human":
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        pygame.init()
        if pygame.display.get_surface() is None:
            pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Valet-Park RL")
        # Human mode draws to the window; otherwise each env keeps its own off-screen
        # scratch surface so several headless envs can share one process safely.
        if render_mode == "human":
            self._surface = pygame.display.get_surface()
        else:
            self._surface = pygame.Surface((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.parkingfont = pygame.font.Font('freesansbold.ttf', 20)
        self.timefont = pygame.font.SysFont('calibri', 30)
        self.background = pygame.image.load('Resources/Map.png').convert()
        self._overlap = pygame.sprite.collide_rect_ratio(0.7)

        # Only spend time drawing when someone will look at the pixels.
        self._draw_visuals = render_mode in ("human", "rgb_array")
        self._present = render_mode == "human"

        self.action_space = spaces.Discrete(6)
        self.obs_size = 5 + self.max_cars * 8
        self.observation_space = spaces.Box(low=-1.0, high=1.0,
                                            shape=(self.obs_size,), dtype=np.float32)

        # Filled in by reset()
        self.car = pygame.sprite.Group()
        self.spots = pygame.sprite.GroupSingle()
        self.player = pygame.sprite.GroupSingle()
        self.clients = []
        self.total_clients = 0
        self.cars_entered = 0
        self.pending = 0
        self.penalty = 0.0
        self.delivered_total = 0
        self.driving = False
        self.car_selected = None
        self.tick = 0
        self._entrance_blocked = False
        self._remaining = self.game_time
        self._shape_prev = None
        self._shape_goal = None

    # ------------------------------------------------------------------ Gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)  # CarSelection uses the stdlib random module

        self.spots = pygame.sprite.GroupSingle()
        self.spots.add(Spots())
        self.player = pygame.sprite.GroupSingle()
        self.player.add(Player())
        self.player.sprite.active = False  # the env drives the player directly
        self.car = pygame.sprite.Group()

        # list(ParkingSpots): CarSelection pops from the list, so pass a fresh copy.
        self.clients = CarSelection(self.game_time, list(ParkingSpots),
                                    self.num_clients, self.num_car_models)
        self.total_clients = len(self.clients)
        self.cars_entered = 0
        self.pending = 0
        self.penalty = 0.0
        self.delivered_total = 0
        self.driving = False
        self.car_selected = None
        self.tick = 0
        self._entrance_blocked = False
        self._remaining = float(self.game_time)
        self._shape_prev = None
        self._shape_goal = None
        return self._get_obs(), self._get_info()

    def step(self, action):
        action = int(action)
        draw = self._draw_visuals
        surface = self._surface

        self.tick += 1
        seconds = self.tick / self.fps

        if draw:
            surface.blit(self.background, (0, 0))

        # Entrance state at the start of the tick (used for arrivals + penalty + display)
        entrance_blocked = pygame.sprite.spritecollideany(self.spots.sprite, self.car) is not None
        if entrance_blocked:
            self.penalty += (1.0 / self.fps) * PENALTY_RATE

        # --- apply the agent's action ---
        if action == SPACE:
            self._toggle_car()
        if self.driving and self.car_selected is not None:
            self._apply_drive(action)

        # --- scheduled arrivals / departures ---
        if self.tick % self.arrival_ticks == 0:
            if (self.cars_entered + self.pending < self.total_clients
                    and int(self.np_random.integers(0, 2)) == 1):
                if entrance_blocked:
                    self.pending += 1
                else:
                    self._spawn_next()
        if self.tick % self.exit_ticks == 0:
            cars_here = self.car.sprites()
            if cars_here and int(self.np_random.integers(0, 2)) == 1:
                cars_here[int(self.np_random.integers(0, len(cars_here)))].ClientExit()

        # --- static visual layers ---
        if draw:
            DrawParkingSpots(21, surface, self.parkingfont)
            self.spots.draw(surface)
            self.car.draw(surface)
            self.player.draw(surface)

        # --- advance the simulation ---
        if not self.driving:
            self._move_player(action)
        before = self.car.sprites()
        self.car.update(surface)  # moves cars, animates clients, handles delivery + kill
        collisions = self._resolve_collisions()
        deliveries = sum(1 for c in before if c.SuccessDelivery)
        self.delivered_total += deliveries

        # Release a deferred arrival now that the entrance is clear.
        entrance_blocked = pygame.sprite.spritecollideany(self.spots.sprite, self.car) is not None
        if self.pending > 0 and not entrance_blocked and self.cars_entered < self.total_clients:
            self._spawn_next()
            self.pending -= 1

        remaining = self.game_time - seconds - self.penalty
        self._entrance_blocked = entrance_blocked
        self._remaining = remaining

        if draw:
            GameTimer(remaining, entrance_blocked, self.timefont, surface)

        # --- reward ---
        reward = STEP_COST
        reward += DELIVERY_REWARD * deliveries
        reward += COLLISION_PENALTY * collisions
        if entrance_blocked:
            reward += BLOCKED_PENALTY
        if self.reward_shaping:
            reward += self._shaping_reward()

        won = self.cars_entered >= self.total_clients and len(self.car) == 0
        time_up = remaining <= 0
        terminated = bool(won)
        truncated = bool(time_up and not won)
        if won:
            reward += WIN_BONUS

        obs = self._get_obs()
        info = self._get_info()

        if self._present:
            pygame.event.pump()  # keep the OS window responsive
            pygame.display.flip()
            self.clock.tick(self.fps)

        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return np.transpose(pygame.surfarray.array3d(self._surface), (1, 0, 2))
        return None  # "human" frames are presented inside step()

    def close(self):
        if pygame.get_init():
            pygame.quit()

    # -------------------------------------------------------------- internals
    def _spawn_next(self):
        car_img_index, person, spot = self.clients[self.cars_entered]
        self.car.add(Car(spot, car_img_index, person))
        self.cars_entered += 1

    def _move_player(self, action):
        p = self.player.sprite
        if action == UP:
            p.rect.y -= PLAYER_SPEED
        elif action == DOWN:
            p.rect.y += PLAYER_SPEED
        elif action == LEFT:
            p.rect.x -= PLAYER_SPEED
        elif action == RIGHT:
            p.rect.x += PLAYER_SPEED
        p.boundaries()

    def _toggle_car(self):
        if not self.driving:
            candidate = pygame.sprite.spritecollideany(self.player.sprite, self.car)
            if candidate is not None:
                self.car_selected = candidate
                candidate.active = True
                self.player.sprite.rect.x = -300  # hide the player while driving
                self.driving = True
        else:
            c = self.car_selected
            c.active = False
            c.activeforward = False
            c.activebackward = False
            c.direction = 0
            self.player.sprite.rect.x = c.rect.x + 30
            self.player.sprite.rect.y = c.rect.y + 70
            self.driving = False
            self.car_selected = None

    def _apply_drive(self, action):
        c = self.car_selected
        c.direction = 1 if action == RIGHT else (-1 if action == LEFT else 0)
        c.activeforward = (action == UP)
        c.activebackward = (action == DOWN)

    def _shaping_reward(self):
        """Potential-based progress of the driven car toward its current goal."""
        c = self.car_selected
        if not self.driving or c is None or c not in self.car:
            self._shape_prev = None
            self._shape_goal = None
            return 0.0
        waiting = c.Client.sprite.ClientExited
        goal = EXIT_TARGET if waiting else self.spot_xy.get(c.ParkingSpot, EXIT_TARGET)
        goal_id = (id(c), waiting)
        dist = math.hypot(c.rect.centerx - goal[0], c.rect.centery - goal[1])
        shaped = 0.0
        if self._shape_prev is not None and goal_id == self._shape_goal:
            shaped = SHAPE_K * (self._shape_prev - dist)
        self._shape_prev = dist
        self._shape_goal = goal_id
        return shaped

    def _resolve_collisions(self):
        """Undo the driven car's move if it overlaps another car; return collision count."""
        sprites = self.car.sprites()
        count = 0
        for i in range(len(sprites)):
            for j in range(i + 1, len(sprites)):
                a, b = sprites[i], sprites[j]
                if self._overlap(a, b):
                    if a.active:
                        a.rect.center = a.prev_center
                        count += 1
                    if b.active:
                        b.rect.center = b.prev_center
                        count += 1
        return count

    def _get_obs(self):
        obs = np.zeros(self.obs_size, dtype=np.float32)
        p = self.player.sprite
        obs[0] = p.rect.centerx / WIDTH
        obs[1] = p.rect.centery / HEIGHT
        obs[2] = 1.0 if self.driving else 0.0
        obs[3] = self._remaining / self.game_time
        obs[4] = 1.0 if self._entrance_blocked else 0.0

        for i, c in enumerate(self.car.sprites()[:self.max_cars]):
            base = 5 + i * 8
            rad = math.radians(c.angle)
            obs[base + 0] = 1.0
            obs[base + 1] = c.rect.centerx / WIDTH
            obs[base + 2] = c.rect.centery / HEIGHT
            obs[base + 3] = math.sin(rad)
            obs[base + 4] = math.cos(rad)
            obs[base + 5] = (c.ParkingSpot - 1001) / 17.0
            obs[base + 6] = 1.0 if c.Client.sprite.ClientExited else 0.0
            obs[base + 7] = 1.0 if c.active else 0.0
        return np.clip(obs, -1.0, 1.0)

    def _get_info(self):
        return {
            "tick": self.tick,
            "time_remaining": max(0.0, self._remaining),
            "cars_present": len(self.car),
            "cars_entered": self.cars_entered,
            "delivered_total": self.delivered_total,
            "pending": self.pending,
            "entrance_blocked": bool(self._entrance_blocked),
        }
