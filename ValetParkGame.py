import pygame
from random import randint

from src.ParkingSpots import DrawParkingSpots
from src.Car import Car
from src.Player import Player
from src.utils import *
from src.config import *


# Issues fixed in this pass:
#  - IndexError: more cars tried to enter than there were clients (cars vs clients mismatch)
#  - UnboundLocalError in the game timer when the entrance-blocked flag was 0
#  - AttributeError: pressing SPACE while not standing on a car (NoneType.active)
#  - Car would not drive after re-entering (space_pressed / Car_select desync)
#  - Entrance-blocked detection was computed inside the event loop (stale/undefined)
#  - Deferred-arrival logic could lose cars / never re-fire
#  - Smoother accumulating time penalty + clear win / lose states
#  - Car-vs-car collisions now block movement instead of passing through
#
# Next milestone (RL): extract this loop into a reset()/step()/observation/reward
# environment so an agent can drive instead of the keyboard. See README.


def resolve_car_collisions(cars):
    """Prevent the driven car from overlapping others by undoing its last move."""
    overlap = pygame.sprite.collide_rect_ratio(0.7)  # 0.7 avoids false hits from rotated bounding boxes
    sprites = cars.sprites()
    for i in range(len(sprites)):
        for j in range(i + 1, len(sprites)):
            a, b = sprites[i], sprites[j]
            if overlap(a, b):
                if a.active:
                    a.rect.center = a.prev_center
                if b.active:
                    b.rect.center = b.prev_center


def spawn_next_car(cars, clients, index):
    """Add the car for clients[index] to the group. Returns the next index."""
    car_img_index, person, spot = clients[index]
    cars.add(Car(spot, car_img_index, person))
    return index + 1


# Initialise pygame and the window
pygame.init()
clock = pygame.time.Clock()
screen = pygame.display.set_mode((1366, 768))  # 720p
parkingfont = pygame.font.Font('freesansbold.ttf', 20)
GameTimeFont = pygame.font.SysFont('calibri', 30)
background = pygame.image.load('Resources/Map.png')

pygame.display.set_caption("Valet-Park")
icon = pygame.image.load('Resources/valet_icon.png')
pygame.display.set_icon(icon)

# Sprite groups
spots = pygame.sprite.GroupSingle()
spots.add(Spots())
player = pygame.sprite.GroupSingle()
player.add(Player())
car = pygame.sprite.Group()

# Game state
ClientsList = CarSelection(GameTime, ParkingSpots, Number_Clients, number_cars_available)
total_clients = len(ClientsList)
car_number = 0           # cars that have entered so far (index into ClientsList)
pending_cars = 0         # arrivals deferred because the entrance was blocked
penalty = 0.0            # extra seconds charged while the entrance is blocked
PENALTY_RATE = 0.2       # blocked time elapses 1.2x as fast
Car_select = False       # is the player currently driving a car?
Car_selected = None      # the car being driven
game_state = "playing"   # "playing" | "won" | "lost"

# Arrival / departure events
Car_enter = pygame.USEREVENT + 1
Car_exit = pygame.USEREVENT + 2
pygame.time.set_timer(Car_enter, 10000)  # every 10s a client may arrive
pygame.time.set_timer(Car_exit, 20000)   # every 20s a client may come to collect

start_ticks = pygame.time.get_ticks()
prev_ticks = start_ticks
running = True

# Game loop
while running:
    now = pygame.time.get_ticks()
    dt = (now - prev_ticks) / 1000.0
    prev_ticks = now
    seconds = (now - start_ticks) / 1000.0

    screen.blit(background, (0, 0))

    # Is a car physically sitting on / driving through the entrance right now?
    entrance_blocked = pygame.sprite.spritecollideany(spots.sprite, car) is not None
    if entrance_blocked and game_state == "playing":
        penalty += dt * PENALTY_RATE

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # A new client may arrive
        if event.type == Car_enter:
            if car_number + pending_cars < total_clients and randint(0, 1) == 1:
                if entrance_blocked:
                    pending_cars += 1
                    print("Entrance blocked - arrival deferred")
                else:
                    car_number = spawn_next_car(car, ClientsList, car_number)

        # A parked client may come out to collect their car
        if event.type == Car_exit:
            cars_here = car.sprites()
            if cars_here and randint(0, 1) == 1:
                cars_here[randint(0, len(cars_here) - 1)].ClientExit()

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE and game_state != "playing":
                running = False

            # Enter / leave the car the player is standing on
            if event.key == pygame.K_SPACE and car:
                if not Car_select:
                    candidate = pygame.sprite.spritecollideany(player.sprite, car)
                    if candidate is not None:
                        Car_selected = candidate
                        player.sprite.active = False
                        Car_selected.active = True
                        player.sprite.rect.x = -300
                        Car_select = True
                else:
                    player.sprite.active = True
                    Car_selected.active = False
                    player.sprite.rect.x = Car_selected.rect.x + 30
                    player.sprite.rect.y = Car_selected.rect.y + 70
                    Car_select = False

            # Drive the active car
            if Car_select:
                if event.key == pygame.K_RIGHT: Car_selected.direction += 1
                if event.key == pygame.K_LEFT: Car_selected.direction -= 1
                if event.key == pygame.K_UP: Car_selected.activeforward = True
                if event.key == pygame.K_DOWN: Car_selected.activebackward = True

        if event.type == pygame.KEYUP and Car_select:
            if event.key == pygame.K_RIGHT: Car_selected.direction -= 1
            if event.key == pygame.K_LEFT: Car_selected.direction += 1
            if event.key == pygame.K_UP: Car_selected.activeforward = False
            if event.key == pygame.K_DOWN: Car_selected.activebackward = False

    # Release one deferred arrival now that the entrance is clear.
    # Recompute here so a car spawned this frame counts as blocking.
    entrance_blocked = pygame.sprite.spritecollideany(spots.sprite, car) is not None
    if pending_cars > 0 and not entrance_blocked and car_number < total_clients:
        car_number = spawn_next_car(car, ClientsList, car_number)
        pending_cars -= 1

    if game_state == "lost":
        screen.fill((0, 0, 0))
        screen.blit(GameTimeFont.render("Game Over", True, (255, 255, 255)), (600, 384))
    elif game_state == "won":
        screen.fill((1, 50, 32))
        screen.blit(GameTimeFont.render("Congrats! You Win", True, (255, 255, 255)), (550, 384))
    else:
        DrawParkingSpots(21, screen, parkingfont)
        spots.draw(screen)
        car.draw(screen)
        player.draw(screen)
        player.update()
        car.update(screen)
        resolve_car_collisions(car)

        remaining = GameTime - seconds - penalty
        time_up = GameTimer(remaining, entrance_blocked, GameTimeFont, screen)

        # Win once every client has arrived and every car has been delivered.
        if car_number >= total_clients and len(car) == 0:
            game_state = "won"
        elif time_up:
            game_state = "lost"

    pygame.display.update()
    clock.tick(60)  # 60 FPS

pygame.quit()
