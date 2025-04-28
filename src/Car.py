from typing import KeysView
import pygame
from .utils import import_folder
import os
from random import randint
from math import sin, radians, degrees, copysign
from pygame.math import Vector2
from .config import *  # Import sound and other settings



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
            # Re-try loading images or create a fallback image
            fallback_image = pygame.Surface((38, 72))
            fallback_image.fill((200, 100, 100))  # Red color for visibility
            self.ClientImg = [fallback_image]
            print(f"Warning: ClientImg is empty for client {self.clientnumber_string}. Using fallback image.")
            return
        
        if current_time - self.last_update > self.animation_speed:
            self.frame = (self.frame + 1) % len(self.ClientImg)
            self.last_update = current_time

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


class Car(pygame.sprite.Sprite):
    def __init__(self, ParkingSpot, CarImgIndex, Clientnumberstr, car_group=None):
        super().__init__()
        self.car_group = car_group
        self._rotated_images = {}
        self._rotated_masks = {}  # Cache for rotated collision masks
        CarImgCyellow = pygame.image.load('Resources/Cars/lr_classic_yellow.png').convert_alpha()
        CarImgCcyan = pygame.image.load('Resources/Cars/lr_classic_cyan.png').convert_alpha()
        CarImgCred = pygame.image.load('Resources/Cars/lr_classic_red.png').convert_alpha()
        CarImgCblue = pygame.image.load('Resources/Cars/lr_classic_blue.png').convert_alpha()
        CarImgCpink = pygame.image.load('Resources/Cars/lr_classic_pink.png').convert_alpha()        
        CarImgCghost = pygame.image.load('Resources/Cars/lr_classic_ghost.png').convert_alpha()
        CarImgMred = pygame.image.load('./Resources/Cars/lr_modern_red.png').convert_alpha()
        CarImgMblue = pygame.image.load('Resources/Cars/lr_modern_blue.png').convert_alpha()
        CarImgMpink = pygame.image.load('Resources/Cars/lr_modern_pink.png').convert_alpha()        
        CarImgMghost = pygame.image.load('Resources/Cars/lr_modern_ghost.png').convert_alpha()
        CarImgSyellow = pygame.image.load('Resources/Cars/lr_super_yellow.png').convert_alpha()
        CarImgSpink = pygame.image.load('Resources/Cars/lr_super_pink.png').convert_alpha()        
        CarImgSghost = pygame.image.load('Resources/Cars/lr_super_ghost.png').convert_alpha()
        self.CarImg = [CarImgCyellow,CarImgCcyan,CarImgCred,CarImgCghost,CarImgCblue,CarImgCpink,CarImgMred,CarImgMghost,CarImgMblue,CarImgMpink,CarImgSyellow,CarImgSghost,CarImgSpink]
        
        # Safety check to ensure CarImgIndex is valid
        try:
            self.CarImgIndex = int(CarImgIndex)
            if self.CarImgIndex < 0 or self.CarImgIndex >= len(self.CarImg):
                print(f"Warning: CarImgIndex {self.CarImgIndex} out of range. Using default index 0.")
                self.CarImgIndex = 0
        except (ValueError, TypeError):
            print(f"Warning: Invalid CarImgIndex {CarImgIndex}. Using default index 0.")
            self.CarImgIndex = 0
            
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
        self.last_position = pygame.math.Vector2(0, 0)
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
        
    def set_active(self, active_state):
        """Safely handle activation/deactivation of the car"""
        if self.active != active_state:  # Only reset if state actually changes
            self.active = active_state
            if not active_state:
                self.reset_controls()

    def handle_collision(self, other_car=None):
        """Enhanced collision handling with directional movement restrictions"""
        if other_car:
            # Calculate offset between the two cars
            offset = (other_car.rect.x - self.rect.x, other_car.rect.y - self.rect.y)
            
            # Check if masks overlap at current offset
            if self.mask.overlap(other_car.mask, offset):
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
                    
                    # If moving into the collision, stop and push back
                    if (self.activeforward and movement_dot > 0) or (self.activebackward and movement_dot < 0):
                        # Zero out velocity
                        self.velocity = pygame.math.Vector2(0, 0)
                        
                        # Store which directions are blocked due to collision
                        # This will be used in accelerate() to prevent movement in blocked directions
                        self.blocked_directions = {
                            'forward': self.activeforward and movement_dot > 0,
                            'backward': self.activebackward and movement_dot < 0,
                            'collision_vector': collision_vector
                        }
                        
                        # Push back slightly to prevent getting stuck
                        push_amount = 5.0
                        self.rect.center = (
                            self.rect.centerx + collision_vector.x * push_amount,
                            self.rect.centery + collision_vector.y * push_amount
                        )
                        
                        # Play collision sound if enabled
                        if ENABLE_SOUND and self.collision_sound:
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
        if self.rect.y == 125 and self.rect.x == 1130:
            if not self.delivery_scored:
                self.delivery_scored = True
                self.rect.x = 1600
                self.Client.sprite.ClientX = 1500
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



    def update(self, screen):
        """Main update function with improved collision response and driving feel"""
        if self.ClientEntered != True:
            self.initial_animation(screen)
            
        if self.active:
            # Store previous position for collision detection and recovery
            prev_pos = pygame.math.Vector2(self.rect.center)
            
            # Update vehicle steering and rotation
            self.set_rotation()
            
            # Apply acceleration and velocity updates
            self.accelerate()
            
            # Check if new position would cause collision using the stored car group
            if self.car_group:
                collided = pygame.sprite.spritecollide(self, self.car_group, False, 
                                                      collided=pygame.sprite.collide_mask)
                if collided:
                    for other in collided:
                        if other != self:
                            # If collision detected, handle the collision properly
                            if self.handle_collision(other):
                                # Show small bounce effect on significant collisions
                                if self.velocity.length() > 0.5:
                                    # Calculate bounce direction
                                    bounce_vector = pygame.math.Vector2(
                                        self.rect.centerx - other.rect.centerx,
                                        self.rect.centery - other.rect.centery
                                    )
                                    if bounce_vector.length() > 0:
                                        bounce_vector.normalize_ip()
                                        # Apply small bounce effect
                                        self.velocity += bounce_vector * COLLISION_BOUNCE
                            break
            
            # Apply boundary constraints
            self.boundaries()
            
            # Draw feedback and visual indicators
            self.draw_parking_feedback(screen)
            
        # Handle client animation and car delivery
        if self.Client.sprite.ClientExited:
            self.Client.sprite.ClientShopExit(screen)
            if not self.active:
                self.successfulDelivery()
                
        # Remove car if delivery successful
        if self.SuccessDelivery == True:
            self.kill()
