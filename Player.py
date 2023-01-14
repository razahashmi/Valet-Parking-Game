import pygame



class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        PlayerImgRight = pygame.image.load('Resources/Players/PlayerRight1.png').convert_alpha()
        PlayerImgLeft = pygame.image.load('Resources/Players/PlayerLeft1.png').convert_alpha()
        PlayerImgUp = pygame.image.load('Resources/Players/PlayerUp1.png').convert_alpha()
        PlayerImgDown = pygame.image.load('Resources/Players/PlayerDown1.png').convert_alpha()
        self.PlayerImg = [PlayerImgRight,PlayerImgLeft,PlayerImgUp,PlayerImgDown]
        self.PlayerImgIndex = 3
        self.active = True
        self.image = pygame.transform.scale(self.PlayerImg[int(self.PlayerImgIndex)], (40, 40))
        self.rect = self.image.get_rect(center = (700,250))

    def player_input(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            self.PlayerImgIndex = 2
            self.image = pygame.transform.scale(self.PlayerImg[int(self.PlayerImgIndex)], (40, 40))
            self.rect.y += -3 
        if keys[pygame.K_DOWN]:
            self.PlayerImgIndex = 3
            self.image = pygame.transform.scale(self.PlayerImg[int(self.PlayerImgIndex)], (40, 40))    
            self.rect.y += 3 
        if keys[pygame.K_RIGHT]:
            self.PlayerImgIndex = 0
            self.image = pygame.transform.scale(self.PlayerImg[int(self.PlayerImgIndex)], (40, 40))        
            self.rect.x += 3 
        if keys[pygame.K_LEFT]:
            self.PlayerImgIndex = 1
            self.image = pygame.transform.scale(self.PlayerImg[int(self.PlayerImgIndex)], (40, 40))      
            self.rect.x += -3 

    def boundaries(self):      
        if self.rect.x <= 90:
            self.rect.x = 90
        elif self.rect.x >= 1250:
            self.rect.x = 1250
        if self.rect.y <= 125:
            self.rect.y = 125
        elif self.rect.y >= 720:
            self.rect.y = 720

    def update(self):
        if self.active:
            self.player_input()
            self.boundaries()