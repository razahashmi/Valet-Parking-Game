import os
import pygame
from random import randint, choice

from src.ParkingSpots import DrawParkingSpots
from src.Car import Car
from src.Player import Player
from src.utils import *
from src.config import *


# Bugs fixed in this pass:
#  - Re-entering a car made it drive/turn by itself: exiting a car left its
#    activeforward / activebackward / direction flags set (and the KEYUP handler
#    is gated on Car_select, so releasing the key could not clear them). Now the
#    drive flags are reset whenever a car is entered or left.
#  - CarSelection was mutating the global ParkingSpots list (shuffle + pop);
#    it now receives a fresh copy.
#  - The game was often unwinnable: pickups fired for a random car at 50%, so
#    many clients were never called. Each pickup event now reliably calls one
#    not-yet-called client, so every client can be delivered within the time.
#
# The per-frame logic lives in ValetGame.step(events, dt) so it can be driven
# headlessly by tests (see tests/) as well as by the real keyboard loop in run().


class ValetGame:
    WIDTH, HEIGHT = 1366, 768
    PENALTY_RATE = 0.2  # blocked time elapses 1.2x as fast

    def __init__(self, present=True, game_time=GameTime, num_clients=Number_Clients):
        self.present = present
        if not present:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        pygame.init()
        if pygame.display.get_surface() is None:
            pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        if present:
            self.screen = pygame.display.get_surface()
            pygame.display.set_caption("Valet-Park")
            try:
                pygame.display.set_icon(pygame.image.load('Resources/valet_icon.png'))
            except pygame.error:
                pass
        else:
            self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))

        self.clock = pygame.time.Clock()
        self.parkingfont = pygame.font.Font('freesansbold.ttf', 20)
        self.GameTimeFont = pygame.font.SysFont('calibri', 30)
        self.background = pygame.image.load('Resources/Map.png').convert()

        self.spots = pygame.sprite.GroupSingle()
        self.spots.add(Spots())
        self.player = pygame.sprite.GroupSingle()
        self.player.add(Player())
        self.car = pygame.sprite.Group()

        self.game_time = game_time
        # list(ParkingSpots): CarSelection pops from the list, so pass a fresh copy.
        self.ClientsList = CarSelection(game_time, list(ParkingSpots),
                                        num_clients, number_cars_available)
        self.total_clients = len(self.ClientsList)
        self.car_number = 0       # cars that have entered so far (index into ClientsList)
        self.pending_cars = 0     # arrivals deferred because the entrance was blocked
        self.penalty = 0.0        # extra seconds charged while the entrance is blocked
        self.Car_select = False   # is the player currently driving a car?
        self.Car_selected = None  # the car being driven
        self.game_state = "playing"   # "playing" | "won" | "lost"
        self.entrance_blocked = False
        self.remaining = float(game_time)
        self.elapsed = 0.0
        self.running = True

        self.Car_enter = pygame.USEREVENT + 1
        self.Car_exit = pygame.USEREVENT + 2

    # ------------------------------------------------------------------ helpers
    def spawn_next_car(self):
        car_img_index, person, spot = self.ClientsList[self.car_number]
        self.car.add(Car(spot, car_img_index, person))
        self.car_number += 1

    def resolve_car_collisions(self):
        """Prevent the driven car from overlapping others by undoing its last move."""
        overlap = pygame.sprite.collide_rect_ratio(0.7)  # avoid false hits from rotated boxes
        sprites = self.car.sprites()
        for i in range(len(sprites)):
            for j in range(i + 1, len(sprites)):
                a, b = sprites[i], sprites[j]
                if overlap(a, b):
                    if a.active:
                        a.rect.center = a.prev_center
                    if b.active:
                        b.rect.center = b.prev_center

    def _enter_car(self, candidate):
        self.Car_selected = candidate
        self.player.sprite.active = False
        candidate.active = True
        # Start from rest so stale flags can never make the car move on its own.
        candidate.activeforward = False
        candidate.activebackward = False
        candidate.direction = 0
        self.player.sprite.rect.x = -300
        self.Car_select = True

    def _leave_car(self):
        c = self.Car_selected
        c.active = False
        # Clear drive flags so the car does not keep moving / turning after we step out
        # (and cannot resume by itself when re-entered).
        c.activeforward = False
        c.activebackward = False
        c.direction = 0
        self.player.sprite.active = True
        self.player.sprite.rect.x = c.rect.x + 30
        self.player.sprite.rect.y = c.rect.y + 70
        self.Car_select = False

    # ------------------------------------------------------------------ events
    def handle_event(self, event):
        if event.type == pygame.QUIT:
            self.running = False
            return

        # A new client may arrive
        if event.type == self.Car_enter:
            if self.car_number + self.pending_cars < self.total_clients and randint(0, 1) == 1:
                if self.entrance_blocked:
                    self.pending_cars += 1
                else:
                    self.spawn_next_car()

        # A parked client comes out to collect their car. Call one client that has
        # arrived but has not been called yet, so every client is eventually served.
        if event.type == self.Car_exit:
            uncalled = [c for c in self.car.sprites()
                        if c.ClientEntered and not c.Client.sprite.ClientExited]
            if uncalled:
                choice(uncalled).ClientExit()

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE and self.game_state != "playing":
                self.running = False

            # Enter / leave the car the player is standing on
            if event.key == pygame.K_SPACE and self.car:
                if not self.Car_select:
                    candidate = pygame.sprite.spritecollideany(self.player.sprite, self.car)
                    if candidate is not None:
                        self._enter_car(candidate)
                else:
                    self._leave_car()

            # Drive the active car
            if self.Car_select:
                if event.key == pygame.K_RIGHT: self.Car_selected.direction += 1
                if event.key == pygame.K_LEFT: self.Car_selected.direction -= 1
                if event.key == pygame.K_UP: self.Car_selected.activeforward = True
                if event.key == pygame.K_DOWN: self.Car_selected.activebackward = True

        if event.type == pygame.KEYUP and self.Car_select:
            if event.key == pygame.K_RIGHT: self.Car_selected.direction -= 1
            if event.key == pygame.K_LEFT: self.Car_selected.direction += 1
            if event.key == pygame.K_UP: self.Car_selected.activeforward = False
            if event.key == pygame.K_DOWN: self.Car_selected.activebackward = False

    # ------------------------------------------------------------------ per frame
    def step(self, events, dt):
        self.elapsed += dt
        seconds = self.elapsed

        self.screen.blit(self.background, (0, 0))

        # Is a car physically sitting on / driving through the entrance right now?
        self.entrance_blocked = pygame.sprite.spritecollideany(self.spots.sprite, self.car) is not None
        if self.entrance_blocked and self.game_state == "playing":
            self.penalty += dt * self.PENALTY_RATE

        for event in events:
            self.handle_event(event)

        # Release one deferred arrival now that the entrance is clear.
        self.entrance_blocked = pygame.sprite.spritecollideany(self.spots.sprite, self.car) is not None
        if self.pending_cars > 0 and not self.entrance_blocked and self.car_number < self.total_clients:
            self.spawn_next_car()
            self.pending_cars -= 1

        if self.game_state == "lost":
            self.screen.fill((0, 0, 0))
            self.screen.blit(self.GameTimeFont.render("Game Over", True, (255, 255, 255)), (600, 384))
        elif self.game_state == "won":
            self.screen.fill((1, 50, 32))
            self.screen.blit(self.GameTimeFont.render("Congrats! You Win", True, (255, 255, 255)), (550, 384))
        else:
            DrawParkingSpots(21, self.screen, self.parkingfont)
            self.spots.draw(self.screen)
            self.car.draw(self.screen)
            self.player.draw(self.screen)
            self.player.update()
            self.car.update(self.screen)
            self.resolve_car_collisions()

            self.remaining = self.game_time - seconds - self.penalty
            time_up = GameTimer(self.remaining, self.entrance_blocked, self.GameTimeFont, self.screen)

            # Win once every client has arrived and every car has been delivered.
            if self.car_number >= self.total_clients and len(self.car) == 0:
                self.game_state = "won"
            elif time_up:
                self.game_state = "lost"

    # ------------------------------------------------------------------ human loop
    def run(self):
        pygame.time.set_timer(self.Car_enter, 10000)  # every 10s a client may arrive
        pygame.time.set_timer(self.Car_exit, 20000)   # every 20s a client is called
        prev_ticks = pygame.time.get_ticks()
        while self.running:
            now = pygame.time.get_ticks()
            dt = (now - prev_ticks) / 1000.0
            prev_ticks = now
            self.step(pygame.event.get(), dt)
            pygame.display.update()
            self.clock.tick(60)  # 60 FPS
        pygame.quit()


def main():
    ValetGame(present=True).run()


if __name__ == "__main__":
    main()
