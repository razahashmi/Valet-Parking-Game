import pygame

# Cache for rendered parking spots
_spot_cache = {}

class SpotCache:
    def __init__(self, rectangles, spots, time):
        self.rectangles = rectangles
        self.spots = spots  # Store the spots for redrawing
        self.last_update = time
    
    def draw(self, screen):
        # Draw all cached spots
        for spot in self.spots:
            screen.blit(spot['surface'], spot['pos'])

def DrawParkingSpots(numSpots, screen, parkingfont):
    global _spot_cache
    current_time = pygame.time.get_ticks()
    
    # Use cached spots if available and not too old
    if numSpots in _spot_cache:
        cache = _spot_cache[numSpots]
        if current_time - cache.last_update < 1000:  # Cache valid for 1 second
            cache.draw(screen)
            return cache.rectangles
    
    spot_rectangles = []
    cached_spots = []
    depthx = 100
    heighty = 150
    initialx = 380
    initialy = 300
    botlefty = 300
    leftspotx = 242
    leftspoty = 300
    rightspotx = 1127
    botrighty = 300
    botinitialx = 380
    spotnumber = 1001
    
    for i in range(0, numSpots):
        spot_surface = pygame.Surface((max(depthx, heighty) + 10, max(depthx, heighty) + 10), pygame.SRCALPHA)
        
        if i <= 5:
            rect = pygame.Rect(initialx, initialy, depthx, heighty)
            spot_rectangles.append({"number": spotnumber, "rect": rect})
            # Draw to cache surface
            pygame.draw.lines(spot_surface, (242, 241, 221), False, 
                            [[0, 0], [0, heighty], [depthx, heighty], [depthx, 0]], 5)
            number = parkingfont.render(str(spotnumber), True, (255, 253, 208))
            spot_surface.blit(number, (25, 120))
            cached_spots.append({'surface': spot_surface, 'pos': (initialx, initialy)})
            spotnumber += 1
            initialx += depthx
            
        elif i <= 11:
            rect = pygame.Rect(botinitialx, 2*initialy-heighty, depthx, heighty)
            spot_rectangles.append({"number": spotnumber, "rect": rect})
            # Draw to cache surface
            pygame.draw.lines(spot_surface, (242, 241, 221), False,
                            [[0, heighty], [0, 0], [depthx, 0], [depthx, heighty]], 5)
            number = parkingfont.render(str(spotnumber), True, (255, 253, 208))
            spot_surface.blit(number, (25, 10))
            cached_spots.append({'surface': spot_surface, 'pos': (botinitialx, 2*initialy-heighty)})
            spotnumber += 1
            botinitialx += depthx
            
        elif i <= 14:
            rect = pygame.Rect(leftspotx-heighty, botlefty, heighty, depthx)
            spot_rectangles.append({"number": spotnumber, "rect": rect})
            # Draw to cache surface rotated 90 degrees
            pygame.draw.lines(spot_surface, (242, 241, 221), False,
                            [[heighty, 0], [0, 0], [0, depthx], [heighty, depthx]], 5)
            number = parkingfont.render(str(spotnumber), True, (255, 253, 208))
            number = pygame.transform.rotate(number, 90)
            spot_surface.blit(number, (10, 30))
            cached_spots.append({'surface': spot_surface, 'pos': (leftspotx-heighty, botlefty)})
            spotnumber += 1
            botlefty += depthx
            
        elif i <= 17:
            rect = pygame.Rect(rightspotx, botrighty, heighty, depthx)
            spot_rectangles.append({"number": spotnumber, "rect": rect})
            # Draw to cache surface
            pygame.draw.lines(spot_surface, (242, 241, 221), False,
                            [[0, 0], [heighty, 0], [heighty, depthx], [0, depthx]], 5)
            number = parkingfont.render(str(spotnumber), True, (255, 253, 208))
            number = pygame.transform.rotate(number, 90)
            spot_surface.blit(number, (heighty-30, 30))
            cached_spots.append({'surface': spot_surface, 'pos': (rightspotx, botrighty)})
            spotnumber += 1
            botrighty += depthx
    
    # Store in cache
    cache = SpotCache(spot_rectangles, cached_spots, current_time)
    _spot_cache[numSpots] = cache
    
    # Draw all spots
    cache.draw(screen)
    
    return spot_rectangles

class Spots(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.image.load('Resources/Entrance_spot.png').convert_alpha()
        self.rect = self.image.get_rect(center = (40,160))

