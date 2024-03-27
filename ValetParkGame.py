import pygame
from random import randint,choice

from pygame.constants import USEREVENT
from ParkingSpots import DrawParkingSpots
from Car import Car
import os
from Player import Player
from Func import *
from config import *
import time


# Issues to resolve:
#  Cars collision function
# bug where if entrance blocked parking lot cars dont drive correctly
# car wont drive once i re enter
# work on the game time logic
# Rework car driving code




# Intialize the pygame
pygame.init()
# all of the game configs
clock = pygame.time.Clock()
screen = pygame.display.set_mode((1366, 768)) # resolution is 720p
parkingfont = pygame.font.Font('freesansbold.ttf', 20)
GameTimeFont = pygame.font.SysFont('calibri', 30)
background = pygame.image.load('Resources/Map.png')
PlayerX = 60 # Player starting position x
PlayerY = 250 # Player starting position y
playerX_change = 0 # change in player position x
playerY_change = 0 # change in player position x
start_ticks=pygame.time.get_ticks()
GameOver = False
space_pressed = 0
parkingfont = pygame.font.Font('freesansbold.ttf', 20)
Gametimer = GameTime #game time set in config. This the overall time of the game
pygame.display.set_caption("Valet-Park") # Caption for Game
icon = pygame.image.load('Resources/valet_icon.png') # Game Icon File
pygame.display.set_icon(icon) # Game Icon
cwd = os.getcwd()
spots = pygame.sprite.GroupSingle()
spots.add(Spots())
Penalize_entrance_blocked = None
running = True
ClientsList = CarSelection(GameTime,ParkingSpots,Number_Clients,number_cars_available)
car = pygame.sprite.Group()
car_entrance_blocked = None
seconds = 0
# add the player in the sprite class
player = pygame.sprite.GroupSingle()
player.add(Player())
Cars_present = []
Car_select = None
car_number = 0
Car_enter = pygame.USEREVENT+1
Car_exit = pygame.USEREVENT+2
pygame.time.set_timer(Car_enter,10000) #time in milliseconds
pygame.time.set_timer(Car_exit,20000) #time in milliseconds
# Game Logic
# add event types that will handle Client arrival and another one for exit. 
# The event will be called every x seconds and will have a probablility y of introducing a new client (if entrance not blocked)
# Similarly for the client exiting (keeping at max 2 exits)  
while running:
    seconds=(pygame.time.get_ticks())/1000
    print(seconds)
    screen.blit(background, (0, 0))
    for event in pygame.event.get():
        # Quit game
        if event.type == pygame.QUIT:
                running = False
        Entrance_blocked = pygame.sprite.spritecollideany(spots.sprite,car)
        # Movement of car object 
        if event.type == Car_enter:
            car_enter_probability = randint(0,1)
            if (car_enter_probability == 1) and (car_number < number_cars_available):
                if not Entrance_blocked:
                    print("car should enter")
                    car.add(Car(ClientsList[car_number][2],ClientsList[car_number][0],ClientsList[car_number][1]))
                    car_number += 1
                if Entrance_blocked:
                    print("Entrance Blocked")
                    car_entrance_blocked = car_number

        if not Entrance_blocked and car_entrance_blocked:
            car.add(Car(ClientsList[car_entrance_blocked][2],ClientsList[car_entrance_blocked][0],ClientsList[car_entrance_blocked][1]))
            car_number += 1
            car_entrance_blocked = None

        if event.type == Car_exit:
            car_exit_probability = randint(0,1)
            number_of_clients = len(car.sprites())
            if car_exit_probability:
                if number_of_clients:
                    print("car should exit")
                    print(number_of_clients)
                    client_selected = randint(0,number_of_clients-1)
                    car_exit = car.sprites()[client_selected]
                    car_exit.ClientExit()


        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                    # Enter/leave car
                    # Active car/player
                if car:
                    if space_pressed == 0:
                        Car_selected = pygame.sprite.spritecollideany(player.sprite,car)
                        player.sprite.active = False
                        Car_selected.active = True
                        # Car_selected.angle = 0
                        # Car_selected.direction = 0


                        player.sprite.rect.x = -300
                        Car_select= True



                    else:
                        if space_pressed == 1 and Car_select:
                            player.sprite.active = True
                            Car_selected.active = False
                            player.sprite.rect.x = Car_selected.rect.x + 30
                            player.sprite.rect.y = Car_selected.rect.y + 70
                            Car_select = False
                    if space_pressed == 1:
                        space_pressed = 0
                    else: space_pressed = 1                
            if Car_select:
                if event.key == pygame.K_RIGHT: Car_selected.direction += 1
                if event.key == pygame.K_LEFT: Car_selected.direction -= 1
                if event.key == pygame.K_UP: Car_selected.activeforward = True
                if event.key == pygame.K_DOWN: Car_selected.activebackward = True
                if event.key == pygame.K_ESCAPE and GameOver: running = False 

        if event.type == pygame.KEYUP:
            if Car_select:
                if event.key == pygame.K_RIGHT: Car_selected.direction -= 1
                if event.key == pygame.K_LEFT: Car_selected.direction += 1
                if event.key == pygame.K_UP: Car_selected.activeforward = False
                if event.key == pygame.K_DOWN: Car_selected.activebackward = False
    if GameOver:
        # GameOver Screen
        screen.fill((0,0,0))
        GameOverRender = GameTimeFont.render("Game Over", True, (255, 255, 255))
        screen.blit(GameOverRender,(600, 384))
    else:
        # Game screen
        DrawParkingSpots(21,screen,parkingfont)
        spots.draw(screen)
        car.draw(screen)
        player.draw(screen)
        player.update()
        car.update(screen)

        GameOver = GameTimer(seconds,GameTime,GameTimeFont,screen,car_entrance_blocked)
        if round(seconds,1) == GameTime and not car:
            # Win screen
            screen.fill((1,50,32))
            GameOverRender = GameTimeFont.render("Congrats! You Win", True, (255, 255, 255))
            screen.blit(GameOverRender,(550, 384))

    pygame.display.update()
    clock.tick(60) # FPS is 60