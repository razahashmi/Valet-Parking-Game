from typing import KeysView
import pygame
from .utils import import_folder
import os
import math  # Add missing math import
from random import randint
from math import sin, radians, degrees, copysign
from pygame.math import Vector2
from .config import *  # Import sound and other settings

# Define missing physics constants that might not be in config.py
if not 'FRICTION' in globals():
    FRICTION = 0.02
if not 'COLLISION_BOUNCE' in globals():
    COLLISION_BOUNCE = 0.3
if not 'REVERSE_MULTIPLIER' in globals():
    REVERSE_MULTIPLIER = 0.6
if not 'ACCELERATION_CURVE' in globals():
    ACCELERATION_CURVE = 0.8
if not 'MIN_SPEED_TO_TURN' in globals():
    MIN_SPEED_TO_TURN = 0.5
if not 'STEERING_LIMIT' in globals():
    STEERING_LIMIT = 1.0
if not 'DRIFT_THRESHOLD' in globals():
    DRIFT_THRESHOLD = 4.0
if not 'COLLISION_SOUND_VOLUME' in globals():
    COLLISION_SOUND_VOLUME = 0.5
if not 'PARK_SOUND_VOLUME' in globals():
    PARK_SOUND_VOLUME = 0.7
if not 'ENABLE_SOUND' in globals():
    ENABLE_SOUND = True



class Client(pygame.sprite.Sprite):
    # Cache for scaled client images
    _scaled_images = {}
    
    def __init__(self, clientnumber_string):
        super().__init__()
        # Convert clientnumber_string to string if it's not already
        self.clientnumber_string = str(clientnumber_string)
        self.import_assets()
        self.setup_animations()
        self.ClientX = 90
        self.ClientY = 50
        self.ClientX_change = 2  # Reduced from 4 to 2 for slower walking
        self.ClientExited = False
        self.last_update = pygame.time.get_ticks()
        self.animation_speed = 150  # Increased from 100 to 150ms for smoother animation
    
    def import_assets(self):
        assets_path = './Resources/Clients'
        cache_key = (self.clientnumber_string, 38, 72)
        
        if cache_key not in Client._scaled_images:
            try:
                full_path = assets_path + "/" + self.clientnumber_string
                raw_images = import_folder(full_path)
                
                # Check if we got any images
                if not raw_images:
                    print(f"Warning: No images found in {full_path}. Using fallback image.")
                    # Create a fallback image
                    fallback_image = pygame.Surface((38, 72))
                    fallback_image.fill((200, 100, 100))  # Red color for visibility
                    Client._scaled_images[cache_key] = [fallback_image]
                else:
                    # Pre-scale all images for better performance
                    Client._scaled_images[cache_key] = [
                        pygame.transform.scale(img, (38, 72))
                        for img in raw_images
                    ]
            except Exception as e:
                print(f"Error loading client images: {e}. Using fallback image.")
                # Create a fallback image
                fallback_image = pygame.Surface((38, 72))
                fallback_image.fill((200, 100, 100))  # Red color for visibility
                Client._scaled_images[cache_key] = [fallback_image]
        
        self.ClientImg = Client._scaled_images[cache_key]

    def setup_animations(self):
        self.frame = 0
    
    def update_animation(self, current_time):
        # Add safety check to prevent ZeroDivisionError
        if not self.ClientImg or len(self.ClientImg) == 0:
            return
            
        # Check if it's time to update the animation
        if current_time - self.last_update > self.animation_speed:
            self.last_update = current_time
            self.frame = (self.frame + 1) % len(self.ClientImg)

    def ClientExit(self):
        if not self.ClientExited:
            self.ClientX = 820
            self.ClientY = 50
            self.ClientExited = True

    def ClientWalkAnimation(self, screen):
        current_time = pygame.time.get_ticks()
        self.update_animation(current_time)
        screen.blit(self.ClientImg[self.frame], (self.ClientX, self.ClientY))

    def ClientShopExit(self, screen):
        if self.ClientX <= 1250:
            # Add pause when reaching the endpoint
            if 1200 <= self.ClientX <= 1210:
                # Pause for 2 seconds (120 frames at 60 FPS)
                if not hasattr(self, '_pause_counter'):
                    self._pause_counter = 120
                if self._pause_counter > 0:
                    self._pause_counter -= 1
                    self.ClientWalkAnimation(screen)
                    return
                
            self.ClientWalkAnimation(screen)
            self.ClientX += self.ClientX_change
        self.ClientWalkAnimation(screen)

    def update(self):
        # Only move the client if it hasn't exited
        if not self.ClientExited:
            self.ClientX += self.ClientX_change
            
        # Use a safety check before accessing ClientImg
        if self.ClientImg and self.frame < len(self.ClientImg):
            self.image = self.ClientImg[self.frame]
        
        # Check if client has exited (moved past exit threshold)
        if self.ClientX >= 1400:
            self.ClientExited = True
            
            # Add safe deletion marker when client completely exits
            if hasattr(self, 'car') and self.car:
                self.car.safe_to_remove = True


class Car(pygame.sprite.Sprite):
    def __init__(self, ParkingSpot, CarImgIndex, Clientnumberstr, car_group=None):
        super().__init__()
        self.car_group = car_group
        self._rotated_images = {}
        self._rotated_masks = {}  # Cache for rotated collision masks
        self.marked_for_deletion = False  # Flag to track if car should be removed
        self.safe_to_remove = False  # Flag that indicates car can be safely removed
        self.delete_cooldown = 0  # Cooldown timer before actual deletion
        self.deletion_timeout = 180  # Frames to wait before deletion (3 seconds at 60fps)
        self.exiting_game = False  # New flag to track if car is exiting the game
        self.position_history = []  # Track recent positions for better stuck detection
        
        # Initialize vectors properly
        self.forward = pygame.math.Vector2(1, 0)  # Default forward vector
        self.right = pygame.math.Vector2(0, 1)    # Default right vector (perpendicular to forward)
        
        # Physics constants for car movement
        self.acceleration_rate = 0.2
        self.max_speed = 7.0
        self.friction = 0.05
        self.steering_rate = 3.0
        
        # Load car images only once - using direct list of images
        self.CarImg = [
            pygame.image.load('Resources/Cars/lr_classic_yellow.png').convert_alpha(),
            pygame.image.load('Resources/Cars/lr_classic_cyan.png').convert_alpha(),
            pygame.image.load('Resources/Cars/lr_classic_red.png').convert_alpha(),
            pygame.image.load('Resources/Cars/lr_classic_ghost.png').convert_alpha(),
            pygame.image.load('Resources/Cars/lr_classic_blue.png').convert_alpha(),
            pygame.image.load('Resources/Cars/lr_classic_pink.png').convert_alpha(),
            pygame.image.load('Resources/Cars/lr_modern_red.png').convert_alpha(),
            pygame.image.load('Resources/Cars/lr_modern_blue.png').convert_alpha(),
            pygame.image.load('Resources/Cars/lr_modern_pink.png').convert_alpha(),
            pygame.image.load('Resources/Cars/lr_modern_ghost.png').convert_alpha(),
            pygame.image.load('Resources/Cars/lr_super_yellow.png').convert_alpha(),
            pygame.image.load('Resources/Cars/lr_super_pink.png').convert_alpha(),
            pygame.image.load('Resources/Cars/lr_super_ghost.png').convert_alpha(),
        ]
        
        # Ensure CarImgIndex is a valid integer between 0 and len(self.CarImg)-1
        try:
            self.CarImgIndex = int(CarImgIndex) % len(self.CarImg)
        except (ValueError, TypeError):
            print(f"Warning: Invalid CarImgIndex '{CarImgIndex}'. Using random color instead.")
            self.CarImgIndex = randint(0, len(self.CarImg)-1)
            
        print(f"Creating car with color index: {self.CarImgIndex}")
        
        self.original_image = pygame.transform.rotate(pygame.transform.scale(self.CarImg[self.CarImgIndex], (70, 150)),-90)
        self.image = self.original_image
        self.mask = pygame.mask.from_surface(self.image)  # Create initial mask
        self.rect = self.image.get_rect(center = (-100,160))
        self.ParkingSpotfont = pygame.font.SysFont('Comic Sans MS', 20)
        self.ParkingSpot = ParkingSpot
        self.assigned_spot = ParkingSpot  # Initialize assigned_spot
        self.velocity = pygame.math.Vector2(0, 0)
        self.acceleration = ACCELERATION
        self.max_speed = MAX_SPEED
        self.brake_force = BRAKE_FORCE
        self.turn_speed = TURN_SPEED
        self.drift_factor = DRIFT_FACTOR
        self.angle = 0
        self.steering_angle = 0
        self.direction = 0
        self.last_position = pygame.math.Vector2(self.rect.center)
        self.is_drifting = False
        self.active = False
        self.ClientEntered = False
        self.forward = pygame.math.Vector2(1, 0)
        self.activeforward = False
        self.activebackward = False
        self.SuccessDelivery = False
        self.entrycar = False
        self.Client = pygame.sprite.GroupSingle()
        self.Client.add(Client(Clientnumberstr))
        self.parked = False
        self.parking_alignment = 0  # 0-100%, how well the car is aligned in the spot
        self.park_start_time = 0
        self.current_spot = None  # Where the car is actually parked
        self.wrong_spot_time = None  # When did we park in wrong spot
        self.has_collided = False  # Track if this car has been in a collision
        self.delivery_scored = False  # Track if delivery points were given
        self.stuck_counter = 0  # Counter to track if car is stuck
        self.last_checked_position = pygame.math.Vector2(self.rect.center)
        self.last_position_check_time = pygame.time.get_ticks()
        self.recovery_attempts = 0  # Track how many times we've tried to recover this car
        self.last_stuck_reset = 0   # Time of last position reset due to being stuck
        self.last_collision_time = 0  # Track when the last collision occurred
        self.position_history = []  # Track recent positions for better stuck detection

        # Load sound effects
        if ENABLE_SOUND:
            try:
                self.collision_sound = pygame.mixer.Sound('Resources/sounds/collision.wav')
                self.park_sound = pygame.mixer.Sound('Resources/sounds/park.wav')
                self.collision_sound.set_volume(COLLISION_SOUND_VOLUME)
                self.park_sound.set_volume(PARK_SOUND_VOLUME)
            except (FileNotFoundError, pygame.error):
                print("Warning: Sound files not found. Game will continue without sound effects.")
                self.collision_sound = None
                self.park_sound = None
        
    def set_rotation(self):
        speed = self.velocity.length()
        
        # Smoother steering logic with less frame-to-frame variation
        if speed > MIN_SPEED_TO_TURN or self.direction != 0:
            # Simple linear relationship between speed and turn rate
            turn_factor = 1.0 - min(0.7, speed / self.max_speed)
            max_turn_rate = self.turn_speed * turn_factor
            
            if self.direction != 0:
                # Calculate target steering with simpler math for smoother transitions
                target_steering = self.direction * STEERING_LIMIT * turn_factor
                
                # Use much smoother interpolation for steering transitions
                # Lower value = smoother but less responsive, higher value = more responsive but choppier
                smooth_factor = 0.1  # Fixed low value for consistent smoothing
                self.steering_angle = pygame.math.lerp(
                    self.steering_angle,
                    target_steering,
                    smooth_factor
                )
                
                # Apply turn rate with more consistent calculations
                turn_amount = self.steering_angle * max_turn_rate * 0.05  # Scale down to avoid large jumps
                
                # Adjust for reverse driving
                if self.activebackward:
                    turn_amount *= -1
                
                # Check for drift conditions with simplified logic
                if speed > DRIFT_THRESHOLD and abs(self.steering_angle) > STEERING_LIMIT * 0.7:
                    self.is_drifting = True
                    # Apply much gentler drift physics
                    side_vector = self.forward.rotate(90)
                    self.velocity = self.velocity * DRIFT_FACTOR + side_vector * (self.direction * 0.02 * speed)
                else:
                    self.is_drifting = False
                
                # Update angle with smoother increments
                self.angle = (self.angle + turn_amount) % 360
        else:
            # Gradually return steering to center when stopped
            self.steering_angle *= 0.9  # Simple decay instead of complex interpolation
            self.is_drifting = False
        
        # Update forward vector for movement
        self.forward = pygame.math.Vector2(1, 0).rotate(-self.angle)
        
        # Cache rotated images for better performance
        angle = int(self.angle)  # Integer angles to reduce cache size and prevent jitter
        
        # Get or create rotated image
        if angle not in self._rotated_images:
            self._rotated_images[angle] = pygame.transform.rotate(self.original_image, angle)
            self._rotated_masks[angle] = pygame.mask.from_surface(self._rotated_images[angle])
        
        # Update sprite with cached image and mask
        old_center = self.rect.center
        self.image = self._rotated_images[angle]
        self.mask = self._rotated_masks[angle]
        self.rect = self.image.get_rect(center=old_center)

    def accelerate(self):
        # Store last position for collision detection
        self.last_position = pygame.math.Vector2(self.rect.center)
        
        # Enhanced acceleration curve with progressive acceleration feel
        current_speed = self.velocity.length()
        speed_ratio = current_speed / self.max_speed
        
        # Progressive acceleration - stronger initial acceleration that tapers off
        # This creates a more responsive feel when starting from rest
        accel_factor = max(0.5, 1.0 - (speed_ratio ** ACCELERATION_CURVE))
        
        # Check if movement is blocked due to collision
        blocked_forward = hasattr(self, 'blocked_directions') and self.blocked_directions.get('forward', False)
        blocked_backward = hasattr(self, 'blocked_directions') and self.blocked_directions.get('backward', False)
        
        # Apply acceleration with improved feel and blockage detection
        if self.activeforward and not blocked_forward:
            # Enhanced forward acceleration with initial boost
            accel_boost = 1.3 if speed_ratio < 0.2 else 1.0  # Extra kick at low speeds
            forward_accel = self.forward * (self.acceleration * accel_factor * accel_boost)
            self.velocity += forward_accel
        elif self.activebackward and not blocked_backward:
            # More controlled reverse acceleration
            reverse_factor = REVERSE_MULTIPLIER * (0.9 + 0.2 * (1 - speed_ratio))
            reverse_accel = self.forward * (self.acceleration * reverse_factor * accel_factor)
            self.velocity -= reverse_accel
        
        # Reset blocked directions after applying acceleration
        if hasattr(self, 'blocked_directions'):
            del self.blocked_directions
        
        # Enhanced speed limiting with progressive falloff
        if current_speed > 0:
            if self.activeforward and current_speed > self.max_speed:
                # Smoother speed capping
                excess_ratio = current_speed / self.max_speed
                falloff_rate = 0.12 * min(3.0, excess_ratio)
                target_speed = max(self.max_speed, current_speed * (1.0 - falloff_rate))
                self.velocity.scale_to_length(target_speed)
                
            elif self.activebackward and current_speed > (self.max_speed * REVERSE_MULTIPLIER):
                # Better reverse speed regulation
                reverse_limit = self.max_speed * REVERSE_MULTIPLIER
                excess_ratio = current_speed / reverse_limit
                falloff_rate = 0.15 * min(2.5, excess_ratio)
                target_speed = max(reverse_limit, current_speed * (1.0 - falloff_rate))
                self.velocity.scale_to_length(target_speed)
            
            # Dynamic friction based on driving conditions
            if self.is_drifting:
                # Reduced friction during drifting for better slide feel
                friction_factor = FRICTION * 0.7
            else:
                # Progressive friction based on speed
                friction_factor = FRICTION * (0.8 + 0.5 * speed_ratio)
            
            # Apply normal movement friction
            self.velocity *= (1.0 - friction_factor)
            
            # Enhanced braking with progressive response
            if (self.activeforward and self.activebackward) or (not self.activeforward and not self.activebackward):
                # Progressive braking - stronger at higher speeds
                brake_strength = self.brake_force * (0.8 + 0.6 * speed_ratio)
                self.velocity *= (1.0 - brake_strength)
        
        # Update position with smoother movement
        new_pos = pygame.math.Vector2(self.rect.center) + self.velocity
        self.rect.center = new_pos

    def reset_controls(self):
        """Reset all control-related states when switching cars"""
        self.velocity = pygame.math.Vector2(0, 0)
        self.steering_angle = 0
        self.direction = 0
        self.activeforward = False
        self.activebackward = False
        self.is_drifting = False
        self.has_collided = False  # Reset collision state
        self.recovery_attempts = 0  # Reset recovery attempts counter
        
    def set_active(self, active_state):
        """Safely handle activation/deactivation of the car"""
        if self.active != active_state:  # Only reset if state actually changes
            self.active = active_state
            if active_state:
                # When activating car, ensure it's in a valid state
                if self.rect.x < -100 or self.rect.x > 1500:
                    # Car is off-screen, move it to a valid position
                    self.rect.center = (randint(300, 700), randint(250, 450))
                    print(f"Repositioned off-screen car to {self.rect.center}")
                    
                # Reset vehicle controls when activating
                self.velocity = pygame.math.Vector2(0, 0)
                self.steering_angle = 0
                self.direction = 0
                self.activeforward = False
                self.activebackward = False
            else:
                self.reset_controls()

    def handle_collision(self, other_car=None):
        """Enhanced collision handling with directional movement restrictions"""
        current_time = pygame.time.get_ticks()
        
        if other_car:
            # Ignore collisions with cars marked for deletion or far off-screen
            if getattr(other_car, 'safe_to_remove', False) or other_car.rect.x > 2000:
                return False
                
            # Prevent collision handling too frequently (prevents rapid bouncing)
            if current_time - self.last_collision_time < 100:  # 100ms cooldown
                return False
                
            # Calculate offset between the two cars
            offset = (other_car.rect.x - self.rect.x, other_car.rect.y - self.rect.y)
            
            # Check if masks overlap at current offset
            if self.mask and other_car.mask and self.mask.overlap(other_car.mask, offset):
                # Calculate collision vector (direction from other car to this car)
                collision_vector = pygame.math.Vector2(
                    self.rect.centerx - other_car.rect.centerx,
                    self.rect.centery - other_car.rect.centery
                )
                
                if collision_vector.length() > 0:
                    collision_vector.normalize_ip()
                    
                    # Get movement direction vector
                    movement_dir = self.velocity if self.velocity.length() > 0 else self.forward
                    if self.activebackward:
                        movement_dir = -movement_dir
                    
                    # Calculate dot product to determine if moving toward collision
                    movement_dot = movement_dir.dot(collision_vector)
                    
                    # Calculate impact force based on speed
                    impact_speed = self.velocity.length()
                    impact_force = min(15.0, impact_speed * 0.8)  # Cap maximum force
                    
                    # If moving into the collision, stop and push back
                    if (self.activeforward and movement_dot > 0) or (self.activebackward and movement_dot < 0):
                        # Reduce velocity based on collision angle
                        self.velocity -= movement_dir * impact_speed * 0.9
                        
                        # Add perpendicular component to create more natural bounces
                        perp_vector = pygame.math.Vector2(-movement_dir.y, movement_dir.x)
                        self.velocity += perp_vector * impact_speed * 0.3
                        
                        # Store which directions are blocked due to collision
                        self.blocked_directions = {
                            'forward': self.activeforward and movement_dot > 0,
                            'backward': self.activebackward and movement_dot < 0,
                            'collision_vector': collision_vector
                        }
                        
                        # Push back to prevent getting stuck
                        push_amount = min(10.0, 5.0 + impact_force * 0.5)
                        self.rect.center = (
                            self.rect.centerx + collision_vector.x * push_amount,
                            self.rect.centery + collision_vector.y * push_amount
                        )
                        
                        # Update collision time
                        self.last_collision_time = current_time
                        self.has_collided = True
                        
                        # Play collision sound if enabled with volume based on impact
                        if ENABLE_SOUND and self.collision_sound and impact_speed > 1.0:
                            volume = min(1.0, impact_speed / 10.0) * COLLISION_SOUND_VOLUME
                            self.collision_sound.set_volume(volume)
                            self.collision_sound.play()
                        
                        return True
                    
        return False

    def handle_successful_park(self):
        if ENABLE_SOUND and self.park_sound and not self.parked:
            self.park_sound.play()
        self.parked = True
        
    def update_parking_status(self, spot_rectangles):
        """Check if car is properly parked in a spot"""
        if self.active:
            # Check all spots, not just assigned spot
            for spot in spot_rectangles:
                spot_rect = spot["rect"]
                if spot_rect.contains(self.rect):
                    self.parking_alignment = self.check_parking_alignment(spot_rect)
                    if self.parking_alignment > 80:  # Well parked
                        if not self.parked:
                            self.handle_successful_park()
                            self.park_start_time = pygame.time.get_ticks() / 1000
                            self.current_spot = spot["number"]
                            if self.current_spot != self.assigned_spot:
                                self.wrong_spot_time = pygame.time.get_ticks() / 1000
                        self.parked = True
                        return
                    
            # If we get here, we're not well parked in any spot
            self.parked = False
            self.current_spot = None
            self.wrong_spot_time = None

    def check_and_recover_position(self):
        """Enhanced position recovery with better stuck detection"""
        current_time = pygame.time.get_ticks()
        
        # Track position history for better stuck detection (keep last 5 positions)
        if len(self.position_history) >= 10:
            self.position_history.pop(0)
        self.position_history.append(pygame.math.Vector2(self.rect.center))
        
        # Check for off-screen or invalid positions
        if (self.rect.centerx < -50 or self.rect.centerx > 1400 or 
            self.rect.centery < -50 or self.rect.centery > 800):
            
            # Only attempt recovery a limited number of times
            if self.recovery_attempts < 3:
                # Move car back to a valid position
                old_pos = self.rect.center
                self.rect.center = (randint(200, 800), randint(200, 500))
                self.velocity = pygame.math.Vector2(0, 0)
                self.reset_controls()
                self.recovery_attempts += 1
                print(f"Recovered car from invalid position {old_pos} to {self.rect.center}")
                return True
            else:
                # If we've tried to recover too many times, mark for deletion
                # but only if this car isn't currently active (being driven)
                if not self.active:
                    print(f"Car recovery failed after {self.recovery_attempts} attempts, marking for deletion")
                    self.marked_for_deletion = True
                    self.delete_cooldown = self.deletion_timeout  # Start deletion timeout
                else:
                    # For active cars, try one more recovery to a safer position
                    self.rect.center = (500, 400)
                    self.velocity = pygame.math.Vector2(0, 0)
                    print("Force-repositioning active car that was off-screen")
        
        # Check if car is stuck in the same position for too long
        if current_time - self.last_position_check_time > 2000:  # Check every 2 seconds
            current_pos = pygame.math.Vector2(self.rect.center)
            distance_moved = current_pos.distance_to(self.last_checked_position)
            
            # Detect if car is stuck by analyzing position history
            if self.active and len(self.position_history) >= 10:
                # Calculate average movement over recent history
                avg_movement = 0
                for i in range(1, len(self.position_history)):
                    avg_movement += self.position_history[i].distance_to(self.position_history[i-1])
                avg_movement /= (len(self.position_history) - 1)
                
                # If active and barely moved, increment stuck counter
                if avg_movement < 3:
                    self.stuck_counter += 1
                    # If stuck for too long (6 checks = ~12 seconds), try to unstick
                    if self.stuck_counter > 6 and current_time - self.last_stuck_reset > 15000:
                        print(f"Car stuck at {self.rect.center} - trying to unstuck")
                        
                        # Apply a small random impulse to try to free the car
                        unstick_angle = randint(0, 359)
                        unstick_vector = pygame.math.Vector2(1, 0).rotate(unstick_angle)
                        self.velocity = unstick_vector * 2.0
                        
                        # If that doesn't work after a few tries, reposition
                        if self.recovery_attempts >= 2:
                            old_pos = self.rect.center
                            # Choose a new safe position away from current
                            self.rect.center = (randint(300, 700), randint(250, 450))
                            self.velocity = pygame.math.Vector2(0, 0)
                            print(f"Repositioned stuck car from {old_pos} to {self.rect.center}")
                        
                        self.stuck_counter = 0
                        self.recovery_attempts += 1
                        self.last_stuck_reset = current_time
                        return True
                else:
                    # Reset counter if car moved
                    self.stuck_counter = max(0, self.stuck_counter - 1)
            
            # Update position tracking
            self.last_checked_position = current_pos
            self.last_position_check_time = current_time
        
        return False

    def boundaries(self):
        # Enhanced boundary collision with more natural bounce and feedback
        boundary_collision = False
        
        # Check x-axis boundaries
        if self.rect.x <= 90:
            self.rect.x = 90
            if self.velocity.x < 0:
                # Apply a gentle bounce effect when hitting walls
                self.velocity.x = -self.velocity.x * COLLISION_BOUNCE * 0.5
                boundary_collision = True
        elif self.rect.x >= 1130:
            self.rect.x = 1130
            if self.velocity.x > 0:
                self.velocity.x = -self.velocity.x * COLLISION_BOUNCE * 0.5
                boundary_collision = True
                
        # Check y-axis boundaries
        if self.rect.y <= 125:
            self.rect.y = 125
            if self.velocity.y < 0:
                self.velocity.y = -self.velocity.y * COLLISION_BOUNCE * 0.5
                boundary_collision = True
        elif self.rect.y >= 610:
            self.rect.y = 610
            if self.velocity.y > 0:
                self.velocity.y = -self.velocity.y * COLLISION_BOUNCE * 0.5
                boundary_collision = True
        
        # Play collision sound if we hit boundary at significant speed
        if boundary_collision and self.velocity.length() > 1.0:
            if ENABLE_SOUND and self.collision_sound:
                # Play at lower volume for wall collisions
                self.collision_sound.set_volume(COLLISION_SOUND_VOLUME * 0.5)
                self.collision_sound.play()
                # Reset volume for future collisions
                self.collision_sound.set_volume(COLLISION_SOUND_VOLUME)
            
            # Apply small speed reduction after wall collision
            self.velocity *= 0.85

    def initial_animation(self, screen):
        if self.rect.x <= 40:
            self.rect.x += 3
        if self.rect.x >= 40:
            self.Client.sprite.ClientX += self.Client.sprite.ClientX_change
            self.Client.sprite.ClientWalkAnimation(screen)
            # Update animation frame using the new system
            current_time = pygame.time.get_ticks()
            if current_time - self.Client.sprite.last_update > self.Client.sprite.animation_speed:
                self.Client.sprite.frame = (self.Client.sprite.frame + 1) % len(self.Client.sprite.ClientImg)
                self.Client.sprite.last_update = current_time
            
            if self.Client.sprite.ClientX <= 300:
                # Only show the target spot number briefly
                if pygame.time.get_ticks() % 3000 < 1500:  # Flash the number every 1.5 seconds
                    self.ParkingSpotRender = self.ParkingSpotfont.render(str(self.ParkingSpot), False, (0, 0, 0))
                    pygame.draw.ellipse(screen, (255,255,255), (self.Client.sprite.ClientX -10, self.Client.sprite.ClientY-45, 60, 40))
                    screen.blit(self.ParkingSpotRender,(self.Client.sprite.ClientX, self.Client.sprite.ClientY -40))
        
        if self.Client.sprite.ClientX >= 700:
            self.Client.sprite.ClientX = -100
            self.Client.sprite.ClientWalkAnimation(screen)
            self.entrycar = False
            self.ClientEntered = True
  

            
    def successfulDelivery(self):
        # Modified to mark car for deletion instead of directly killing it
        if self.rect.y <= 125 and self.rect.x >= 1130:
            if not self.delivery_scored:
                self.delivery_scored = True
                self.marked_for_deletion = True  # Mark for controlled deletion
                self.SuccessDelivery = True
                return True  # Return True if points should be awarded
        return False

    def ClientExit(self):
        if self.ClientEntered:
            self.Client.sprite.ClientExit()
    

    def check_parking_alignment(self, spot_rect):
        # Check if car is within parking spot bounds
        if not spot_rect.contains(self.rect):
            return 0
        
        # Calculate alignment percentage based on angle and position
        spot_center = spot_rect.center
        car_center = self.rect.center
        distance = pygame.math.Vector2(spot_center).distance_to(pygame.math.Vector2(car_center))
        
        # Calculate angle difference (assuming parking spots are aligned to 90 degrees)
        target_angle = 90  # This should match the spot's orientation
        angle_diff = abs((self.angle % 360) - target_angle)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
            
        # Perfect parking is when car is centered and aligned
        position_score = max(0, 1 - (distance / (spot_rect.width / 2)))
        angle_score = max(0, 1 - (angle_diff / 45))  # 45 degrees tolerance
        
        return int((position_score * 0.6 + angle_score * 0.4) * 100)

    def draw_parking_feedback(self, screen):
        if self.active:
            # Enhanced visual feedback for better driving experience
            speed = round(self.velocity.length(), 1)
            
            # Draw drift indicators when drifting
            if self.is_drifting:
                # Visual drift effect - orange trail 
                drift_color = (255, 165, 0)  # Orange
                drift_radius = 4 + int(speed)
                drift_pos = (self.rect.centerx, self.rect.centery)
                
                # Draw drift particles
                pygame.draw.circle(screen, drift_color, drift_pos, drift_radius)
                
                # Draw smaller drift particles behind the car
                back_vector = -self.forward * 20
                back_pos = (int(self.rect.centerx + back_vector.x), 
                           int(self.rect.centery + back_vector.y))
                pygame.draw.circle(screen, drift_color, back_pos, drift_radius-2)
            
            # Draw direction indicator (shows which way the car is pointing)
            if speed > 0.5:
                # Direction line shows where car is heading
                direction_length = 25 + (speed * 5)
                start_pos = self.rect.center
                
                # Use velocity direction for moving, forward vector when still
                direction = self.velocity.normalize() if speed > 0.5 else self.forward
                end_pos = (int(start_pos[0] + direction.x * direction_length),
                          int(start_pos[1] + direction.y * direction_length))
                
                # Color changes based on forward/reverse
                if self.activeforward:
                    line_color = (50, 205, 50)  # Green for forward
                elif self.activebackward:
                    line_color = (220, 20, 60)  # Red for reverse
                else:
                    line_color = (200, 200, 200)  # Gray for neutral
                
                # Draw direction indicator
                pygame.draw.line(screen, line_color, start_pos, end_pos, 2)
            
            # Draw alignment feedback with improved visuals
            if self.parking_alignment > 0:
                # Gradient color from red->yellow->green based on alignment
                color = (
                    min(255, 510 - 5.1 * self.parking_alignment),  # Red component
                    min(255, 5.1 * self.parking_alignment),        # Green component
                    0
                )
                
                # Only show text overlay for high alignment values
                if self.parking_alignment > 60:
                    text = f"Alignment: {self.parking_alignment}%"
                    font = pygame.font.SysFont('calibri', 20)
                    text_surface = font.render(text, True, color)
                    screen.blit(text_surface, (self.rect.centerx - 50, self.rect.top - 30))
                    
                    # Add parking guide rectangle when close to aligned
                    pygame.draw.rect(screen, color, self.rect, 2)
                
            # Enhanced speedometer with better visual feedback
            speed_ratio = min(1.0, speed / self.max_speed)
            
            # Change speedometer color based on speed
            if speed_ratio < 0.5:
                speed_color = (100, 255, 100)  # Green for low speed
            elif speed_ratio < 0.8:
                speed_color = (255, 255, 100)  # Yellow for medium speed
            else:
                speed_color = (255, 100, 100)  # Red for high speed
                
            speed_text = f"Speed: {speed}"
            speed_surface = pygame.font.SysFont('calibri', 16).render(speed_text, True, speed_color)
            screen.blit(speed_surface, (self.rect.centerx - 25, self.rect.bottom + 5))

            # Enhanced spot indicator when parked
            if self.parked:
                if self.current_spot == self.assigned_spot:
                    # Correct spot
                    spot_color = (100, 255, 100)
                    spot_text = f"Spot: {self.current_spot} ✓"
                else:
                    # Wrong spot
                    spot_color = (255, 100, 100)
                    spot_text = f"Spot: {self.current_spot} ×"
                    
                spot_surface = pygame.font.SysFont('calibri', 16).render(spot_text, True, spot_color)
                screen.blit(spot_surface, (self.rect.centerx - 35, self.rect.bottom + 25))

    def update(self, dt, cars=None):
        """Update car state"""
        # If car is marked for deletion, handle fade out effect
        if self.marked_for_deletion:
            # Gradually fade out the car
            self.deletion_timer += dt
            if self.deletion_timer >= self.deletion_cooldown:
                self.safe_to_remove = True
                return
            
            # Apply fade effect
            alpha = 255 * (1 - (self.deletion_timer / self.deletion_cooldown))
            self.image.set_alpha(int(alpha))
            return  # Don't process movement for cars being deleted

        # Store position history for stuck detection
        current_time = pygame.time.get_ticks()
        if len(self.position_history) > 20:
            self.position_history.pop(0)  # Remove oldest position
        self.position_history.append((pygame.math.Vector2(self.rect.center), current_time))
        
        # Check if stuck and try to recover
        self.handle_stuck_detection()
        
        # Apply acceleration/deceleration
        if self.activeforward:  # Changed from self.accelerating to self.activeforward
            self.speed += self.acceleration * dt
        elif self.activebackward:  # Changed from self.braking to self.activebackward
            self.speed -= self.brake_force * dt
        else:
            # Slow down due to friction when neither accelerating nor braking
            if self.speed > 0:
                self.speed = max(0, self.speed - self.friction * dt)
            elif self.speed < 0:
                self.speed = min(0, self.speed + self.friction * dt)
        
        # Limit speed
        self.speed = max(-self.max_reverse_speed, min(self.speed, self.max_speed))
        
        # Calculate drift factor based on speed and steering angle
        drift_factor = min(1.0, abs(self.speed) / (self.max_speed * 0.5))
        steering_intensity = abs(self.steering) / self.max_steering
        
        # Apply steering (turn car based on steering direction)
        if abs(self.speed) > 0.1:  # Only turn if moving
            # At higher speeds, steering has more influence on drift
            # At lower speeds, car turns more sharply
            effective_steering = self.steering * (1.0 - (drift_factor * 0.7))
            turn_amount = effective_steering * (abs(self.speed) / self.max_speed) * dt
            self.angle += turn_amount
            self._recalculate_vectors()
        
        # Calculate velocity based on current angle and speed
        self.velocity = self.forward * self.speed
        
        # Update position
        if abs(self.speed) > 0.05:  # Only update position if moving
            new_pos = pygame.math.Vector2(self.rect.center) + self.velocity * dt
            self.rect.center = (new_pos.x, new_pos.y)
        
        # Check for collisions with other cars if provided
        if cars:
            for other_car in cars:
                if other_car != self and self.rect.colliderect(other_car.rect):
                    self.handle_collision(other_car)
                    
        # Check screen boundaries and bounce if needed
        screen_width, screen_height = pygame.display.get_surface().get_size()
        
        # Left and right boundaries
        if self.rect.left < 0:
            self.rect.left = 0
            self.velocity.x = abs(self.velocity.x) * 0.3  # Bounce with reduced velocity
            self.speed *= 0.5  # Reduce speed on collision
        elif self.rect.right > screen_width:
            self.rect.right = screen_width
            self.velocity.x = -abs(self.velocity.x) * 0.3  # Bounce with reduced velocity
            self.speed *= 0.5  # Reduce speed on collision
            
        # Top and bottom boundaries
        if self.rect.top < 0:
            self.rect.top = 0
            self.velocity.y = abs(self.velocity.y) * 0.3  # Bounce with reduced velocity
            self.speed *= 0.5  # Reduce speed on collision
        elif self.rect.bottom > screen_height:
            self.rect.bottom = screen_height
            self.velocity.y = -abs(self.velocity.y) * 0.3  # Bounce with reduced velocity
            self.speed *= 0.5  # Reduce speed on collision

    def draw_fade_effect(self, screen):
        """Draw fade effect for cars being removed"""
        if self.marked_for_deletion and not self.safe_to_remove:
            # Calculate fade based on deletion countdown
            fade_ratio = self.delete_cooldown / self.deletion_timeout
            
            # Draw expanding circle effect
            radius = int(50 * fade_ratio)
            alpha = max(0, 255 - int(255 * fade_ratio))
            
            # Create a surface for the circle with alpha
            circle_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(circle_surface, (255, 255, 255, alpha), (radius, radius), radius)
            
            # Draw circle centered on car
            screen.blit(circle_surface, (self.rect.centerx - radius, self.rect.centery - radius))
            
            # Add fadeout effect to car sprite
            if fade_ratio > 0.5:  # Start fadeout halfway through deletion countdown
                self.image.set_alpha(int(255 * (1 - (fade_ratio - 0.5) * 2)))
    
    def on_key_down(self, key):
        """Process key down events for car control"""
        if self.active:
            if key == pygame.K_UP:
                self.activeforward = True
            elif key == pygame.K_DOWN:
                self.activebackward = True
            elif key == pygame.K_LEFT:
                self.direction = 1
            elif key == pygame.K_RIGHT:
                self.direction = -1
    
    def on_key_up(self, key):
        """Process key up events for car control"""
        if self.active:
            if key == pygame.K_UP:
                self.activeforward = False
            elif key == pygame.K_DOWN:
                self.activebackward = False
            elif key == pygame.K_LEFT and self.direction == 1:
                self.direction = 0
            elif key == pygame.K_RIGHT and self.direction == -1:
                self.direction = 0

    def handle_stuck_detection(self):
        """Detect if car is stuck and attempt recovery if needed"""
        # Ensure we have enough position history
        if len(self.position_history) < 5:
            return False
            
        # Check if position hasn't changed significantly in the last few updates
        stuck_threshold = 5  # pixels
        time_threshold = 2000  # 2 seconds
        
        # Get current position and time
        current_pos, current_time = self.position_history[-1]
        oldest_pos, oldest_time = self.position_history[0]
        
        # Calculate distance moved
        distance = pygame.math.Vector2(current_pos).distance_to(oldest_pos)
        time_elapsed = current_time - oldest_time
        
        # If car hasn't moved much but has been trying to (non-zero speed)
        is_stuck = distance < stuck_threshold and time_elapsed > time_threshold and abs(self.speed) > 0.5
        
        if is_stuck:
            # Initialize recovery attempts counter if needed
            if not hasattr(self, 'recovery_attempts'):
                self.recovery_attempts = 0
                self.last_recovery_time = 0
            
            # Limit recovery frequency and attempts
            if (current_time - self.last_recovery_time > 3000 and 
                self.recovery_attempts < 5):
                self.last_recovery_time = current_time
                self.recovery_attempts += 1
                self._attempt_recovery()
                return True
                
        return False
        
    def _attempt_recovery(self):
        """Try to recover from stuck position"""
        # First, try reversing direction
        self.speed = -self.speed * 1.2
        
        # Add a small random angle change to escape
        import random
        angle_change = random.uniform(-30, 30)
        self.angle += angle_change
        self._recalculate_vectors()
        
        print(f"Car {self.ParkingSpot} attempted recovery from stuck position.")
        
    def _recalculate_vectors(self):
        """Update forward and right vectors based on current angle"""
        # Convert angle to radians (pygame uses degrees)
        angle_rad = math.radians(self.angle)
        
        # Calculate forward vector (unit vector in direction of car's heading)
        self.forward = pygame.math.Vector2(math.cos(angle_rad), math.sin(angle_rad))
        
        # Calculate right vector (perpendicular to forward)
        self.right = pygame.math.Vector2(-self.forward.y, self.forward.x)
        
    def draw(self, screen):
        """Draw the car with opacity based on deletion state"""
        if not self.safe_to_remove:
            # Create a copy of the original image for rotation and alpha
            rotated = pygame.transform.rotate(self.image, -self.angle)
            
            # Apply fade effect if marked for deletion
            if self.marked_for_deletion and hasattr(self, 'delete_cooldown'):
                # Calculate alpha based on remaining cooldown
                alpha_factor = max(0, min(255, int(255 * self.delete_cooldown / self.deletion_timeout)))
                
                # Create a copy with alpha
                temp = rotated.copy()
                temp.set_alpha(alpha_factor)
                rotated = temp
            
            # Calculate position for rotated image
            new_rect = rotated.get_rect(center=self.rect.center)
            screen.blit(rotated, new_rect.topleft)
            
            # Draw additional indicators for debugging (optional)
            if self.active and self.show_debug_info:
                # Draw velocity vector
                if self.velocity.length() > 0:
                    end_pos = (
                        int(self.rect.centerx + self.velocity.normalize() * 30).x,
                        int(self.rect.centery + self.velocity.normalize() * 30).y
                    )
                    pygame.draw.line(screen, (0, 255, 0), self.rect.center, end_pos, 2)
                
                # Draw parking indicator if applicable
                if self.is_parked:
                    pygame.draw.circle(screen, (0, 255, 0), self.rect.center, 10, 2)
                elif self.ParkingSpot is not None:
                    pygame.draw.circle(screen, (255, 255, 0), self.rect.center, 10, 2)
