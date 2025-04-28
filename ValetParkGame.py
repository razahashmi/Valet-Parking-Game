import pygame
from random import randint,choice
from pygame.constants import USEREVENT
from src.ParkingSpots import DrawParkingSpots, Spots
from src.Car import Car
import os
from src.Player import Player
from src.utils import *
from src.config import *  # This imports all settings including ENABLE_SOUND
from src.rl_agent import ValetRLAgent, ActionMapper, calculate_reward
from enum import Enum
import numpy as np
import time

# Global variables
car = None  # Define car as a global variable

class GameState(Enum):
    MENU = 1
    PLAYING = 2
    PAUSED = 3
    GAME_OVER = 4
    WIN = 5
    RL_TRAINING = 6  # New state for RL training mode
    RL_PLAYING = 7   # New state for trained RL agent playing

# Issues to resolve:
#  Cars collision function
# bug where if entrance blocked parking lot cars dont drive correctly
# car wont drive once i re enter
# work on the game time logic
# Rework car driving code


def get_cell(pos, cell_size=100):
    """Get grid cell for position"""
    return (int(pos[0] // cell_size), int(pos[1] // cell_size))

def check_car_collisions():
    """Enhanced collision detection that properly prevents cars from overlapping"""
    penalty_applied = False
    cell_size = 100  # Size of grid cells
    grid = {}  # Spatial hash grid
    
    # Place cars in grid for efficient collision checking
    for car_sprite in car.sprites():
        # Use integer division to get grid cell
        cell = get_cell(car_sprite.rect.center, cell_size)
        if cell not in grid:
            grid[cell] = []
        grid[cell].append(car_sprite)
        
        # Add to neighboring cells too for more reliable detection
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue  # Skip the current cell
                neighbor = (cell[0] + dx, cell[1] + dy)
                if neighbor not in grid:
                    grid[neighbor] = []
                if car_sprite not in grid[neighbor]:  # Avoid duplicates
                    grid[neighbor].append(car_sprite)
    
    # Check collisions between cars in same or adjacent cells
    checked_pairs = set()
    for cell in grid:
        cars = grid[cell]
        for i, car1 in enumerate(cars):
            for car2 in cars[i+1:]:
                # Create an ID for this pair so we don't check twice
                pair_id = tuple(sorted([id(car1), id(car2)]))
                if pair_id in checked_pairs:
                    continue
                    
                checked_pairs.add(pair_id)
                
                # Skip collision check between inactive static cars
                if not car1.active and not car2.active:
                    continue
                
                # Perform more accurate mask-based collision detection
                if pygame.sprite.collide_mask(car1, car2):
                    # When collision detected, handle collision for both cars
                    collision_happened = False
                    
                    # The car that's moving should handle the collision
                    if car1.active or car1.velocity.length() > 0.1:
                        if car1.handle_collision(car2):
                            collision_happened = True
                            
                    # Also let the second car handle collision
                    if car2.active or car2.velocity.length() > 0.1:
                        if car2.handle_collision(car1):
                            collision_happened = True
                    
                    if collision_happened:
                        penalty_applied = True
                        
                        # Force cars apart if they're still overlapping
                        offset = (car2.rect.x - car1.rect.x, car2.rect.y - car1.rect.y)
                        if car1.mask.overlap(car2.mask, offset):
                            # Calculate separation vector
                            sep_vector = pygame.math.Vector2(
                                car1.rect.centerx - car2.rect.centerx, 
                                car1.rect.centery - car2.rect.centery
                            )
                            
                            # Only separate if there's a valid direction
                            if sep_vector.length() > 0:
                                sep_vector.normalize_ip()
                                
                                # Push cars apart more effectively
                                push_amount = 5.0
                                if car1.active and not car2.active:
                                    # Active car hits parked car - push only active car
                                    car1.rect.center = (
                                        car1.rect.centerx + sep_vector.x * push_amount * 2,
                                        car1.rect.centery + sep_vector.y * push_amount * 2
                                    )
                                elif car2.active and not car1.active:
                                    # Active car hits parked car - push only active car
                                    car2.rect.center = (
                                        car2.rect.centerx - sep_vector.x * push_amount * 2,
                                        car2.rect.centery - sep_vector.y * push_amount * 2
                                    )
                                else:
                                    # Both or neither car active - push both
                                    car1.rect.center = (
                                        car1.rect.centerx + sep_vector.x * push_amount,
                                        car1.rect.centery + sep_vector.y * push_amount
                                    )
                                    car2.rect.center = (
                                        car2.rect.centerx - sep_vector.x * push_amount,
                                        car2.rect.centery - sep_vector.y * push_amount
                                    )
    
    return penalty_applied


def main():
    # Make car variable global
    global car
    
    # Initialize pygame
    pygame.init()
    pygame.mixer.init()
    
    # Initialize display with hardware acceleration and double buffering
    screen = pygame.display.set_mode((1366, 768), pygame.HWSURFACE | pygame.DOUBLEBUF)
    
    # Target frame time in milliseconds (60 FPS)
    TARGET_FPS = 60
    FRAME_TIME = 1000 // TARGET_FPS

    # Load sound effects
    if ENABLE_SOUND:
        try:
            game_over_sound = pygame.mixer.Sound('Resources/sounds/game_over.wav')
            win_sound = pygame.mixer.Sound('Resources/sounds/win.wav')
            game_over_sound.set_volume(GAME_OVER_SOUND_VOLUME)
            win_sound.set_volume(WIN_SOUND_VOLUME)
        except (FileNotFoundError, pygame.error):
            print("Warning: Sound files not found. Game will continue without sound effects.")
            game_over_sound = None
            win_sound = None

    # all of the game configs
    clock = pygame.time.Clock()
    parkingfont = pygame.font.Font('freesansbold.ttf', 20)
    GameTimeFont = pygame.font.SysFont('calibri', 30)
    # Cache surfaces and fonts
    parking_spot_cache = {}
    background = pygame.image.load('Resources/Map.png').convert()  # Convert for faster blitting
    overlay = pygame.Surface((1366, 768))
    overlay.set_alpha(128)
    overlay.fill((0, 0, 0))

    PlayerX = 60 # Player starting position x
    PlayerY = 250 # Player starting position y
    playerX_change = 0 # change in player position x
    playerY_change = 0 # change in player position x
    # Initialize game time tracking
    start_ticks = pygame.time.get_ticks()
    seconds = 0
    GameOver = False
    space_pressed = 0
    parkingfont = pygame.font.Font('freesansbold.ttf', 20)
    Gametimer = GameTime #game time set in config. This the overall time of the game
    car_entrance_blocked = None  # Track if a car was blocked from entering
    pygame.display.set_caption("Valet-Park") # Caption for Game
    icon = pygame.image.load('Resources/valet_icon.png') # Game Icon File
    pygame.display.set_icon(icon) # Game Icon
    cwd = os.getcwd()
    # Create sprite groups
    car = pygame.sprite.Group()
    spots = pygame.sprite.Group()
    spots.add(Spots())  # Add entrance spot immediately after creating the group
    player = pygame.sprite.GroupSingle()

    # Initialize player
    player_sprite = Player()
    player_sprite.rect.x = PlayerX
    player_sprite.rect.y = PlayerY
    player.add(player_sprite)

    Penalize_entrance_blocked = None
    running = True
    ClientsList = CarSelection(GameTime,ParkingSpots,Number_Clients,number_cars_available)
    Car_select = None
    car_number = 0
    Car_enter = pygame.USEREVENT+1
    Car_exit = pygame.USEREVENT+2
    pygame.time.set_timer(Car_enter,10000) #time in milliseconds
    pygame.time.set_timer(Car_exit,20000) #time in milliseconds
    # Game Logic
    current_state = GameState.MENU  # Start with the menu
    score = 0  # Initialize score at the right scope
    high_score = 0
    last_delivery_time = {}  # Track delivery times for bonus points
    spot_rectangles = []
    difficulty_level = 1  # 1: Easy, 2: Medium, 3: Hard
    blocked_entrance_penalty_applied = False  # Track if entrance blocked penalty was applied
    spot_cache = None

    # Reinforcement Learning setup
    rl_state_size = 112  # 2 for player + 10 cars * 11 properties per car
    rl_agent = ValetRLAgent(rl_state_size)
    rl_current_episode = 0
    rl_max_episodes = 1000
    rl_successful_parks = 0
    rl_successful_deliveries = 0
    rl_previous_state = None
    rl_current_action = None
    rl_training = False
    rl_collision_occurred = False
    rl_frame_skip = 3  # Only take actions every n frames
    rl_frame_count = 0

    class ScoreDisplay:
        def __init__(self):
            self.last_score = None
            self.surface = None

    score_display = ScoreDisplay()

    def reset_game():
        nonlocal score, car_number, seconds, car_entrance_blocked, current_state, start_ticks
        nonlocal Car_select, space_pressed, Car_selected
        score = 0
        car_number = 0
        seconds = 0
        car_entrance_blocked = None
        Car_select = False
        space_pressed = 0
        Car_selected = None
        car.empty()
        spots.empty()  # Clear existing spots
        spots.add(Spots())  # Add entrance spot
        
        # Reset player position
        player_sprite.active = True
        player_sprite.rect.x = PlayerX
        player_sprite.rect.y = PlayerY
        
        start_ticks = pygame.time.get_ticks()  # Reset start time when game resets

    def adjust_difficulty():
        nonlocal difficulty_level
        global GameTime, QUICK_DELIVERY_THRESHOLD
        if difficulty_level == 1:  # Easy
            GameTime = 300
            QUICK_DELIVERY_THRESHOLD = 30
        elif difficulty_level == 2:  # Medium
            GameTime = 240
            QUICK_DELIVERY_THRESHOLD = 25
        elif difficulty_level == 3:  # Hard
            GameTime = 180
            QUICK_DELIVERY_THRESHOLD = 20

    def draw_menu():
        screen.fill((0, 0, 50))
        title = GameTimeFont.render("Valet Parking", True, SCORE_TEXT_COLOR)
        start_text = GameTimeFont.render("Press ENTER to Start", True, SCORE_TEXT_COLOR)
        rl_train_text = GameTimeFont.render("Press T to Train AI", True, SCORE_TEXT_COLOR)
        rl_play_text = GameTimeFont.render("Press A to Watch AI Play", True, SCORE_TEXT_COLOR)
        diff_text = GameTimeFont.render(f"Difficulty: {'Easy' if difficulty_level == 1 else 'Medium' if difficulty_level == 2 else 'Hard'}", True, SCORE_TEXT_COLOR)
        diff_help = GameTimeFont.render("Press D to change difficulty", True, SCORE_TEXT_COLOR)
        screen.blit(title, (550, 200))
        screen.blit(start_text, (530, 300))
        screen.blit(rl_train_text, (530, 350))
        screen.blit(rl_play_text, (530, 400))
        screen.blit(diff_text, (530, 450))
        screen.blit(diff_help, (530, 500))

    def draw_pause_menu():
        pause_text = GameTimeFont.render("PAUSED", True, SCORE_TEXT_COLOR)
        resume_text = GameTimeFont.render("Press P to Resume", True, SCORE_TEXT_COLOR)
        screen.blit(pause_text, (600, 300))
        screen.blit(resume_text, (550, 400))

    def draw_rl_training_info():
        episode_text = GameTimeFont.render(f"Training Episode: {rl_current_episode}/{rl_max_episodes}", True, SCORE_TEXT_COLOR)
        parks_text = GameTimeFont.render(f"Successful Parks: {rl_successful_parks}", True, SCORE_TEXT_COLOR)
        deliveries_text = GameTimeFont.render(f"Successful Deliveries: {rl_successful_deliveries}", True, SCORE_TEXT_COLOR)
        reward_text = GameTimeFont.render(f"Episode Reward: {rl_agent.total_reward:.1f}", True, SCORE_TEXT_COLOR)
        epsilon_text = GameTimeFont.render(f"Exploration Rate: {rl_agent.epsilon:.2f}", True, SCORE_TEXT_COLOR)
        
        screen.blit(episode_text, (50, 50))
        screen.blit(parks_text, (50, 80))
        screen.blit(deliveries_text, (50, 110))
        screen.blit(reward_text, (50, 140))
        screen.blit(epsilon_text, (50, 170))

    def draw_parking_guide(car):
        if SHOW_PARKING_GUIDE and car.active:
            guide_length = 100
            start_pos = car.rect.center
            end_pos = (start_pos[0] + guide_length * car.forward.x,
                      start_pos[1] + guide_length * car.forward.y)
            pygame.draw.line(screen, GUIDE_LINE_COLOR, start_pos, end_pos, 2)

    def update_score(points, x, y):
        nonlocal score  # Use nonlocal instead of global since score is defined in main()
        score += points
        if points > 0:
            color = SCORE_TEXT_COLOR
        else:
            color = PENALTY_TEXT_COLOR
        score_text = GameTimeFont.render(f"{points:+d}", True, color)
        screen.blit(score_text, (x, y))

    def spawn_new_car():
        nonlocal car_number  # Use nonlocal for car_number as well
        if car_number < number_cars_available:
            # Check if there's already a car near the entrance
            for existing_car in car.sprites():
                if existing_car.rect.centerx < 200:  # Check if any car is in entrance area
                    return False  # Don't spawn if entrance area is occupied
            
            clientnumstr, parkingspot, carindex = ClientsList[car_number]
            new_car = Car(ParkingSpot=parkingspot, CarImgIndex=carindex, Clientnumberstr=clientnumstr, car_group=car)
            car.add(new_car)
            car_number += 1
            return True
        return False
        
    # Initialize variables for RL agent
    Car_selected = None
    rl_previous_state = None
    
    # Process RL agent actions
    def process_rl_action(action):
        nonlocal Car_select, Car_selected, space_pressed, rl_collision_occurred
        
        # Handle different actions
        if action == 4:  # SPACE - enter/exit car
            # Enter/leave car logic
            if car:
                if not Car_select:
                    # Find the nearest car to the player
                    nearest_car = None
                    min_distance = float('inf')
                    for car_sprite in car.sprites():
                        if car_sprite.ClientEntered:  # Only consider cars that are ready to be driven
                            distance = ((player_sprite.rect.centerx - car_sprite.rect.centerx)**2 + 
                                      (player_sprite.rect.centery - car_sprite.rect.centery)**2)**0.5
                            if distance < min_distance:
                                nearest_car = car_sprite
                                min_distance = distance
                    
                    # Enter the nearest car if it's close enough
                    if nearest_car and min_distance < 100:  # Only enter if within reasonable distance
                        player_sprite.active = False
                        nearest_car.set_active(True)
                        nearest_car.reset_controls()
                        player_sprite.rect.x = -300  # Move player off-screen
                        Car_select = True
                        Car_selected = nearest_car
                        space_pressed = 1
                else:
                    # Exit the car
                    player_sprite.active = True
                    Car_selected.set_active(False)
                    player_sprite.rect.x = Car_selected.rect.x + 30
                    player_sprite.rect.y = Car_selected.rect.y + 70
                    Car_select = False
                    space_pressed = 0
                    Car_selected = None
        else:
            # Handle movement controls
            if Car_select and Car_selected:
                # Handle car controls
                if action == 0:  # UP
                    Car_selected.activeforward = True
                    Car_selected.activebackward = False
                elif action == 1:  # DOWN
                    Car_selected.activeforward = False
                    Car_selected.activebackward = True
                elif action == 2:  # LEFT
                    Car_selected.direction = -1
                elif action == 3:  # RIGHT
                    Car_selected.direction = 1
                else:  # Do nothing
                    Car_selected.activeforward = False
                    Car_selected.activebackward = False
                    Car_selected.direction = 0
            else:
                # Handle player movement
                player_speed = 3
                if action == 0:  # UP
                    player_sprite.PlayerImgIndex = 2
                    player_sprite.image = pygame.transform.scale(player_sprite.PlayerImg[int(player_sprite.PlayerImgIndex)], (40, 40))
                    player_sprite.rect.y += -player_speed
                elif action == 1:  # DOWN
                    player_sprite.PlayerImgIndex = 3
                    player_sprite.image = pygame.transform.scale(player_sprite.PlayerImg[int(player_sprite.PlayerImgIndex)], (40, 40))
                    player_sprite.rect.y += player_speed
                elif action == 2:  # LEFT
                    player_sprite.PlayerImgIndex = 1
                    player_sprite.image = pygame.transform.scale(player_sprite.PlayerImg[int(player_sprite.PlayerImgIndex)], (40, 40))
                    player_sprite.rect.x += -player_speed
                elif action == 3:  # RIGHT
                    player_sprite.PlayerImgIndex = 0
                    player_sprite.image = pygame.transform.scale(player_sprite.PlayerImg[int(player_sprite.PlayerImgIndex)], (40, 40))
                    player_sprite.rect.x += player_speed

    # Game loop
    while True:
        # Fixed timestep game loop
        current_time = pygame.time.get_ticks()
        frame_time = clock.tick_busy_loop(TARGET_FPS)
        delta_time = min(frame_time / 1000.0, 0.1)
        seconds = current_time / 1000.0
        
        # Event handling and state updates
        for event in pygame.event.get():
            # Quit game
            if event.type == pygame.QUIT:
                # Save RL agent state if training
                if current_state in [GameState.RL_TRAINING, GameState.RL_PLAYING]:
                    rl_agent.save()
                pygame.quit()
                return
            
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                # Save RL agent state if training
                if current_state in [GameState.RL_TRAINING, GameState.RL_PLAYING]:
                    rl_agent.save()
                pygame.quit()
                return

            # Check if entrance is blocked using the first parking spot sprite
            entrance_spots = spots.sprites()
            Entrance_blocked = False
            if entrance_spots:
                Entrance_blocked = pygame.sprite.spritecollideany(entrance_spots[0], car)
            
            # Movement of car object 
            if event.type == Car_enter:
                car_enter_probability = randint(0,1)
                if (car_enter_probability == 1) and (car_number < number_cars_available):
                    entrance_spots = spots.sprites()
                    Entrance_blocked = False
                    if entrance_spots:
                        Entrance_blocked = pygame.sprite.spritecollideany(entrance_spots[0], car)
                    if not Entrance_blocked:
                        new_car = Car(ClientsList[car_number][2], ClientsList[car_number][0], ClientsList[car_number][1])
                        # Check if spawn position is clear
                        temp_group = pygame.sprite.Group(new_car)
                        if not pygame.sprite.groupcollide(temp_group, car, False, False):
                            car.add(new_car)
                            car_number += 1
                        else:
                            car_entrance_blocked = car_number
                    else:
                        print("Entrance Blocked")
                        car_entrance_blocked = car_number

            # Try to spawn blocked car if entrance is now clear
            if car_entrance_blocked is not None:
                entrance_spots = spots.sprites()
                Entrance_blocked = False
                if entrance_spots:
                    Entrance_blocked = pygame.sprite.spritecollideany(entrance_spots[0], car)
                if not Entrance_blocked:
                    new_car = Car(ClientsList[car_entrance_blocked][2], ClientsList[car_entrance_blocked][0], ClientsList[car_entrance_blocked][1])
                    temp_group = pygame.sprite.Group(new_car)
                    if not pygame.sprite.groupcollide(temp_group, car, False, False):
                        car.add(new_car)
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

            # Menu and state change keys
            if event.type == pygame.KEYDOWN:
                # Handle mode selection in menu
                if current_state == GameState.MENU:
                    if event.key == pygame.K_RETURN:
                        current_state = GameState.PLAYING
                        reset_game()
                    elif event.key == pygame.K_t:
                        current_state = GameState.RL_TRAINING
                        reset_game()
                        rl_training = True
                        rl_current_episode = 0
                        rl_successful_parks = 0
                        rl_successful_deliveries = 0
                    elif event.key == pygame.K_a:
                        # Try to load trained agent
                        if rl_agent.load():
                            current_state = GameState.RL_PLAYING
                            reset_game()
                            rl_training = False
                        else:
                            # Display error message if no trained agent found
                            print("No trained agent found. Please train the agent first.")
                    elif event.key == pygame.K_d:
                        difficulty_level = (difficulty_level % 3) + 1
                        adjust_difficulty()
                
                # Pause/resume handling
                if event.key == pygame.K_p:
                    if current_state == GameState.PLAYING:
                        current_state = GameState.PAUSED
                    elif current_state == GameState.PAUSED:
                        current_state = GameState.PLAYING
                    elif current_state == GameState.RL_TRAINING:
                        current_state = GameState.PAUSED
                    elif current_state == GameState.PAUSED and rl_training:
                        current_state = GameState.RL_TRAINING
                    elif current_state == GameState.RL_PLAYING:
                        current_state = GameState.PAUSED
                    elif current_state == GameState.PAUSED and not rl_training:
                        current_state = GameState.RL_PLAYING
                
                # Return to menu
                if event.key == pygame.K_m:
                    if current_state in [GameState.GAME_OVER, GameState.WIN, 
                                        GameState.RL_TRAINING, GameState.RL_PLAYING]:
                        # Save RL agent state if training
                        if current_state == GameState.RL_TRAINING:
                            rl_agent.save()
                        current_state = GameState.MENU
                
                # Manual player controls only in PLAYING state
                if current_state == GameState.PLAYING:
                    if event.key == pygame.K_SPACE:
                        # Enter/leave car logic
                        if car:
                            if space_pressed == 0:
                                Car_selected = pygame.sprite.spritecollideany(player.sprite, car)
                                if Car_selected:  # Only proceed if we found a car
                                    player.sprite.active = False
                                    Car_selected.set_active(True)
                                    Car_selected.reset_controls()  # Reset controls when entering car
                                    player.sprite.rect.x = -300
                                    Car_select = True
                                    space_pressed = 1
                            else:
                                if Car_select:
                                    player.sprite.active = True
                                    Car_selected.set_active(False)
                                    player.sprite.rect.x = Car_selected.rect.x + 30
                                    player.sprite.rect.y = Car_selected.rect.y + 70
                                    Car_select = False
                                    space_pressed = 0

                    # Move car control handling here, outside of KEYDOWN event
                    if Car_select and Car_selected:  # Check both conditions
                        keys = pygame.key.get_pressed()  # Get current keyboard state
                        if keys[pygame.K_RIGHT]: Car_selected.direction = 1
                        elif keys[pygame.K_LEFT]: Car_selected.direction = -1
                        else: Car_selected.direction = 0
                        Car_selected.activeforward = keys[pygame.K_UP]
                        Car_selected.activebackward = keys[pygame.K_DOWN]

                if event.key == pygame.K_ESCAPE and GameOver: running = False 

            if event.type == pygame.KEYUP and current_state == GameState.PLAYING:
                if Car_select:
                    if event.key == pygame.K_RIGHT: Car_selected.direction = 0
                    if event.key == pygame.K_LEFT: Car_selected.direction = 0
                    if event.key == pygame.K_UP: Car_selected.activeforward = False
                    if event.key == pygame.K_DOWN: Car_selected.activebackward = False
                    
        # Additional key handling outside the event loop for more reliable detection
        keys = pygame.key.get_pressed()
        if current_state == GameState.MENU and keys[pygame.K_RETURN]:
            print("Enter key detected in direct key check")
            reset_game()
            current_state = GameState.PLAYING
        
        # Game over condition check
        if GameOver and current_state in [GameState.PLAYING, GameState.RL_TRAINING, GameState.RL_PLAYING]:
            if current_state == GameState.PLAYING:
                current_state = GameState.GAME_OVER
            elif current_state == GameState.RL_TRAINING:
                # End training episode
                rl_agent.end_episode(rl_current_episode, rl_successful_parks, rl_successful_deliveries)
                rl_current_episode += 1
                rl_successful_parks = 0
                rl_successful_deliveries = 0
                
                # Check if we've reached max episodes
                if rl_current_episode >= rl_max_episodes:
                    rl_agent.save()  # Save the trained agent
                    current_state = GameState.MENU
                    print("RL training completed!")
                else:
                    # Reset for next episode
                    reset_game()
                    rl_previous_state = None
            elif current_state == GameState.RL_PLAYING:
                current_state = GameState.MENU
                
        # Win condition check
        if round(seconds,1) == GameTime and not car and current_state in [GameState.PLAYING, GameState.RL_TRAINING, GameState.RL_PLAYING]:
            if current_state == GameState.PLAYING:
                current_state = GameState.WIN
            elif current_state == GameState.RL_TRAINING:
                # Provide a big bonus for finishing the game
                rl_agent.total_reward += 500
                
                # End training episode with success
                rl_agent.end_episode(rl_current_episode, rl_successful_parks, rl_successful_deliveries)
                rl_current_episode += 1
                rl_successful_parks = 0
                rl_successful_deliveries = 0
                
                # Check if we've reached max episodes
                if rl_current_episode >= rl_max_episodes:
                    rl_agent.save()  # Save the trained agent
                    current_state = GameState.MENU
                    print("RL training completed successfully!")
                else:
                    # Reset for next episode
                    reset_game()
                    rl_previous_state = None
            elif current_state == GameState.RL_PLAYING:
                current_state = GameState.MENU
            
        # Clear screen once per frame
        screen.blit(background, (0, 0))
        
        # Update game state based on current_state
        if current_state == GameState.MENU:
            draw_menu()
        elif current_state == GameState.PAUSED:
            screen.blit(overlay, (0, 0))
            draw_pause_menu()
        elif current_state in [GameState.PLAYING, GameState.RL_TRAINING, GameState.RL_PLAYING]:
            # Common game state updates for all playing modes
            entrance_spots = spots.sprites()
            Entrance_blocked = False
            if entrance_spots:
                Entrance_blocked = pygame.sprite.spritecollideany(entrance_spots[0], car)
            
            # Check for and handle car collisions
            collision_occurred = check_car_collisions()
            rl_collision_occurred = collision_occurred  # Store for RL reward calculation
            
            # Update parking spots
            current_time = pygame.time.get_ticks()
            spot_rectangles = DrawParkingSpots(21, screen, parkingfont)
            
            # Draw game elements
            spots.draw(screen)
            car.draw(screen)
            player.draw(screen)
            
            # Spawn new car if needed (with different probabilities for training)
            if rl_training and randint(0, 100) < 5:  # 5% chance per frame during training
                spawn_new_car()
            
            # For RL agent control
            if current_state in [GameState.RL_TRAINING, GameState.RL_PLAYING]:
                # Get current state
                current_state_vector = rl_agent.get_state(player_sprite, car.sprites(), None, spot_rectangles)
                
                # Every few frames, choose and execute an action
                rl_frame_count += 1
                if rl_frame_count >= rl_frame_skip:
                    rl_frame_count = 0
                    
                    # Choose an action
                    if rl_previous_state is not None:
                        # Calculate reward from previous action
                        reward = calculate_reward(
                            Car_selected, 
                            rl_previous_state, 
                            current_state_vector,
                            rl_current_action,
                            spot_rectangles,
                            rl_collision_occurred
                        )
                        
                        # Process step in RL agent
                        done = GameOver or current_state == GameState.WIN
                        rl_agent.step(rl_previous_state, rl_current_action, reward, current_state_vector, done)
                    
                    # Choose next action
                    rl_current_action = rl_agent.act(current_state_vector, training=(current_state == GameState.RL_TRAINING))
                    process_rl_action(rl_current_action)
                    rl_previous_state = current_state_vector
                
                # Show training info
                if current_state == GameState.RL_TRAINING:
                    draw_rl_training_info()
            
            # Update only active entities
            if current_state == GameState.PLAYING:
                player.update()
            for car_sprite in car.sprites():
                if car_sprite.active or car_sprite.ClientEntered != True or car_sprite.Client.sprite.ClientExited:
                    car_sprite.update(screen)
                    car_sprite.update_parking_status(spot_rectangles)
                    
                    # Check for newly parked car (for RL reward)
                    if car_sprite.parked and not hasattr(car_sprite, '_was_parked'):
                        car_sprite._was_parked = True
                        if car_sprite.current_spot == car_sprite.assigned_spot:
                            rl_successful_parks += 1
            
            # Score and UI updates (only when needed)
            if current_state == GameState.PLAYING:
                if score != score_display.last_score:
                    score_display.surface = GameTimeFont.render(f"Score: {score}", True, SCORE_TEXT_COLOR)
                    score_display.last_score = score
                screen.blit(score_display.surface, (50, 50))
            
            # Draw parking guide only for active car
            if Car_select:
                draw_parking_guide(Car_selected)
            
            # Check for successful delivery
            for car_sprite in car.sprites():
                if car_sprite.successfulDelivery():
                    delivery_time = seconds - last_delivery_time.get(car_sprite.ParkingSpot, seconds)
                    points = POINTS_SUCCESSFUL_PARK
                    
                    if delivery_time < QUICK_DELIVERY_THRESHOLD:
                        points += POINTS_QUICK_DELIVERY
                        
                    # In human-played mode, update visible score
                    if current_state == GameState.PLAYING:
                        update_score(points, car_sprite.rect.centerx, car_sprite.rect.centery - 50)
                        
                    last_delivery_time[car_sprite.ParkingSpot] = seconds
                    
                    # For RL training/playing
                    if current_state in [GameState.RL_TRAINING, GameState.RL_PLAYING]:
                        rl_successful_deliveries += 1
                    
            # Handle entrance blocking penalty
            if car_entrance_blocked and not blocked_entrance_penalty_applied:
                if current_state == GameState.PLAYING:
                    update_score(POINTS_BLOCKED_ENTRANCE_PENALTY, 200, 50)
                blocked_entrance_penalty_applied = True
            elif not car_entrance_blocked:
                blocked_entrance_penalty_applied = False  # Reset when entrance is cleared

            GameOver = GameTimer(seconds, GameTime, GameTimeFont, screen, car_entrance_blocked)

        elif current_state == GameState.GAME_OVER:
            if ENABLE_SOUND and game_over_sound and not getattr(current_state, 'sound_played', False):
                game_over_sound.play()
                current_state.sound_played = True
            # GameOver Screen
            screen.fill((0,0,0))
            GameOverRender = GameTimeFont.render("Game Over", True, (255, 255, 255))
            score_text = GameTimeFont.render(f"Final Score: {score}", True, (255, 255, 255))
            high_score_text = GameTimeFont.render(f"High Score: {high_score}", True, (255, 255, 255))
            restart_text = GameTimeFont.render("Press ENTER to Restart", True, (255, 255, 255))
            menu_text = GameTimeFont.render("Press M for Menu", True, (255, 255, 255))
            screen.blit(GameOverRender,(600, 300))
            screen.blit(score_text, (580, 350))
            screen.blit(high_score_text, (580, 400))
            screen.blit(restart_text, (530, 450))
            screen.blit(menu_text, (550, 500))

        elif current_state == GameState.WIN:
            if ENABLE_SOUND and win_sound and not getattr(current_state, 'sound_played', False):
                win_sound.play()
                current_state.sound_played = True
            # Win screen
            screen.fill((1,50,32))
            GameOverRender = GameTimeFont.render("Congrats! You Win", True, (255, 255, 255))
            score_text = GameTimeFont.render(f"Final Score: {score}", True, (255, 255, 255))
            high_score_text = GameTimeFont.render(f"High Score: {high_score}", True, (255, 255, 255))
            restart_text = GameTimeFont.render("Press ENTER to Play Again", True, (255, 255, 255))
            menu_text = GameTimeFont.render("Press M for Menu", True, (255, 255, 255))
            screen.blit(GameOverRender,(550, 300))
            screen.blit(score_text, (580, 350))
            screen.blit(high_score_text, (580, 400))
            screen.blit(restart_text, (530, 450))
            screen.blit(menu_text, (550, 500))

        pygame.display.flip()  # More efficient than update() for full screen changes
        remaining = FRAME_TIME - pygame.time.get_ticks() % FRAME_TIME
        if remaining > 0:
            pygame.time.wait(remaining)

if __name__ == '__main__':
    main()