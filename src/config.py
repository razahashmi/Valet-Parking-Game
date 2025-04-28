# Game Settings
Number_Clients = 10 # number of Clients
number_cars_available = 12 # total number cars available in the game
GameTime = 300 # Total Game time
ParkingSpots = [1001,1002,1003,1004,1005,1006,1007,1008,1009,1010,1011,1012,1013,1014,1015,1016,1017,1018]

# Vehicle Physics - Enhanced for better driving feel
ACCELERATION = 0.15        # Increased for more responsive acceleration
MAX_SPEED = 3.0            # Slightly higher top speed for better control
BRAKE_FORCE = 0.18         # Stronger brakes for better stopping
TURN_SPEED = 1.5           # Improved turning response
DRIFT_FACTOR = 0.92        # Slight drift effect for more fun driving
REVERSE_MULTIPLIER = 0.6   # Adjusted reverse speed
STEERING_LIMIT = 40        # Higher steering angle for better maneuverability  
MIN_SPEED_TO_TURN = 0.03   # Lower threshold to allow turning at slower speeds
FRICTION = 0.04            # Reduced friction for smoother movement
STEERING_SMOOTHING = 0.2   # More responsive steering
COLLISION_BOUNCE = 0.25    # Better collision response
COLLISION_FORCE = 4.0      # More balanced collision force
COLLISION_PUSH_FORCE = 2.5 # Refined collision push
MIN_SEPARATION = 12.0      # Adjusted separation for better collision avoidance
DRIFT_THRESHOLD = 1.8      # Speed threshold for drift effect
TRACTION_FALLOFF = 0.85    # How much traction reduces during turns (lower = more drift)
ACCELERATION_CURVE = 0.7   # Acceleration curve factor (lower = more initial punch)

# Sound Settings
ENABLE_SOUND = True
COLLISION_SOUND_VOLUME = 0.7
PARK_SOUND_VOLUME = 0.8
GAME_OVER_SOUND_VOLUME = 0.5
WIN_SOUND_VOLUME = 0.8

# Scoring System
POINTS_SUCCESSFUL_PARK = 100
POINTS_QUICK_DELIVERY = 50  # Bonus for delivering under time threshold
POINTS_COLLISION_PENALTY = -30
POINTS_BLOCKED_ENTRANCE_PENALTY = -20
QUICK_DELIVERY_THRESHOLD = 30  # seconds

# Visual Effects
SHOW_PARKING_GUIDE = True
SHOW_TARGET_SPOT = True
COLLISION_EFFECT_DURATION = 0.5  # seconds
GUIDE_LINE_COLOR = (0, 255, 0)  # Green
TARGET_SPOT_COLOR = (255, 255, 0)  # Yellow
PENALTY_TEXT_COLOR = (255, 0, 0)  # Red
SCORE_TEXT_COLOR = (255, 255, 255)  # White
