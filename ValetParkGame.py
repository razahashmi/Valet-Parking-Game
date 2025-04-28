import pygame
from random import randint,choice
from pygame.constants import USEREVENT
from src.ParkingSpots import DrawParkingSpots, Spots
from src.Car import Car
import os
from src.Player import Player
from src.utils import *
from src.config import *  # This imports all settings including ENABLE_SOUND
from enum import Enum

# Global variables
car = None  # Define car as a global variable

class GameState(Enum):
    MENU = 1
    PLAYING = 2
    PAUSED = 3
    GAME_OVER = 4
    WIN = 5

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
    current_state = GameState.PLAYING  # Changed from MENU to PLAYING
    score = 0  # Initialize score at the right scope
    high_score = 0
    last_delivery_time = {}  # Track delivery times for bonus points
    spot_rectangles = []
    difficulty_level = 1  # 1: Easy, 2: Medium, 3: Hard
    blocked_entrance_penalty_applied = False  # Track if entrance blocked penalty was applied
    spot_cache = None

    class ScoreDisplay:
        def __init__(self):
            self.last_score = None
            self.surface = None

    score_display = ScoreDisplay()

    def reset_game():
        nonlocal score, car_number, seconds, car_entrance_blocked, current_state, start_ticks
        score = 0
        car_number = 0
        seconds = 0
        car_entrance_blocked = None
        car.empty()
        spots.empty()  # Clear existing spots
        spots.add(Spots())  # Add entrance spot
        current_state = GameState.PLAYING  # Explicitly set to PLAYING state
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
        diff_text = GameTimeFont.render(f"Difficulty: {'Easy' if difficulty_level == 1 else 'Medium' if difficulty_level == 2 else 'Hard'}", True, SCORE_TEXT_COLOR)
        diff_help = GameTimeFont.render("Press D to change difficulty", True, SCORE_TEXT_COLOR)
        screen.blit(title, (550, 250))
        screen.blit(start_text, (530, 350))
        screen.blit(diff_text, (530, 450))
        screen.blit(diff_help, (530, 500))

    def draw_pause_menu():
        pause_text = GameTimeFont.render("PAUSED", True, SCORE_TEXT_COLOR)
        resume_text = GameTimeFont.render("Press P to Resume", True, SCORE_TEXT_COLOR)
        screen.blit(pause_text, (600, 300))
        screen.blit(resume_text, (550, 400))

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
                pygame.quit()
                return
            
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
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


            if event.type == pygame.KEYDOWN:
                # Handle RETURN key with explicit debug print
                if event.key == pygame.K_RETURN:
                    print(f"Enter key pressed. Current state: {current_state}")
                    if current_state in [GameState.MENU, GameState.GAME_OVER, GameState.WIN]:
                        reset_game()
                        print("Game reset called")
                    
                if event.key == pygame.K_d and current_state == GameState.MENU:
                    difficulty_level = (difficulty_level % 3) + 1
                    adjust_difficulty()

                if event.key == pygame.K_p and current_state == GameState.PLAYING:
                    current_state = GameState.PAUSED
                elif event.key == pygame.K_p and current_state == GameState.PAUSED:
                    current_state = GameState.PLAYING
                    
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

            if event.type == pygame.KEYUP:
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
            
        # Clear screen once per frame
        screen.blit(background, (0, 0))
        
        # Update game state based on current_state
        if current_state == GameState.MENU:
            draw_menu()
        elif current_state == GameState.PAUSED:
            screen.blit(overlay, (0, 0))
            draw_pause_menu()
        elif current_state == GameState.PLAYING:
            # Update game state
            entrance_spots = spots.sprites()
            Entrance_blocked = False
            if entrance_spots:
                Entrance_blocked = pygame.sprite.spritecollideany(entrance_spots[0], car)
            
            # Check for and handle car collisions
            check_car_collisions()
            
            # Update parking spots
            current_time = pygame.time.get_ticks()
            spot_rectangles = DrawParkingSpots(21, screen, parkingfont)
            
            # Draw game elements
            spots.draw(screen)
            car.draw(screen)
            player.draw(screen)
            
            # Update only active entities
            player.update()
            for car_sprite in car.sprites():
                if car_sprite.active or car_sprite.ClientEntered != True or car_sprite.Client.sprite.ClientExited:
                    car_sprite.update(screen)
                    car_sprite.update_parking_status(spot_rectangles)  # Use spot_rectangles directly
            
            # Score and UI updates (only when needed)
            if score != score_display.last_score:
                score_display.surface = GameTimeFont.render(f"Score: {score}", True, SCORE_TEXT_COLOR)
                score_display.last_score = score
            screen.blit(score_display.surface, (50, 50))
            
            # Draw parking guide only for active car
            if Car_select:
                draw_parking_guide(Car_selected)
            
            # Check for successful delivery
            for car_sprite in car.sprites():
                if car_sprite.successfulDelivery():  # Now returns True only on first successful delivery
                    delivery_time = seconds - last_delivery_time.get(car_sprite.ParkingSpot, seconds)
                    points = POINTS_SUCCESSFUL_PARK
                    if delivery_time < QUICK_DELIVERY_THRESHOLD:
                        points += POINTS_QUICK_DELIVERY
                    update_score(points, car_sprite.rect.centerx, car_sprite.rect.centery - 50)
                    last_delivery_time[car_sprite.ParkingSpot] = seconds
                    
            # Handle entrance blocking penalty
            if car_entrance_blocked and not blocked_entrance_penalty_applied:
                update_score(POINTS_BLOCKED_ENTRANCE_PENALTY, 200, 50)
                blocked_entrance_penalty_applied = True
            elif not car_entrance_blocked:
                blocked_entrance_penalty_applied = False  # Reset when entrance is cleared

            GameOver = GameTimer(seconds,GameTime,GameTimeFont,screen,car_entrance_blocked)
            if GameOver:
                current_state = GameState.GAME_OVER
                if score > high_score:
                    high_score = score

            if round(seconds,1) == GameTime and not car:
                # Win screen
                current_state = GameState.WIN
                if score > high_score:
                    high_score = score

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
            screen.blit(GameOverRender,(600, 384))
            screen.blit(score_text, (580, 350))
            screen.blit(high_score_text, (580, 400))
            screen.blit(restart_text, (530, 450))

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
            screen.blit(GameOverRender,(550, 384))
            screen.blit(score_text, (580, 350))
            screen.blit(high_score_text, (580, 400))
            screen.blit(restart_text, (530, 450))

        pygame.display.flip()  # More efficient than update() for full screen changes
        remaining = FRAME_TIME - pygame.time.get_ticks() % FRAME_TIME
        if remaining > 0:
            pygame.time.wait(remaining)

if __name__ == '__main__':
    main()