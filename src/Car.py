from typing import KeysView
import pygame
from .utils import import_folder
import os
from random import randint
from math import sin, radians, degrees, copysign
from pygame.math import Vector2



class Client(pygame.sprite.Sprite):
    def __init__(self,clientnumber_string):
        super().__init__()
        self.import_assets()

        ClientImg1 = self.clientimg_assets[clientnumber_string][0]
        ClientImg2 = self.clientimg_assets[clientnumber_string][1]
        ClientImg3 = self.clientimg_assets[clientnumber_string][2]
        self.ClientImg = [ClientImg1,ClientImg2,ClientImg3]
        self.ClientImgIndex = 0
        self.ClientX = 90
        self.ClientY = 50
        self.ClientX_change = 2
        self.ClientExited = False
    
    def import_assets(self):
        assets_path = './Resources/Clients'
        self.clientimg_assets = { 'person1':[],'person2':[],'person3':[],'person4':[],'person5':[],'person6':[],'person7':[],'person8':[],'person9':[],'person10':[],'person11':[],'person12':[],'person13':[],'person14':[],'person15':[],'person16':[],'person17':[]}
        for clientimgs in self.clientimg_assets.keys():
            full_path = assets_path +"/" +clientimgs
            self.clientimg_assets[clientimgs] = import_folder(full_path)
        

    def ClientExit(self):
        if self.ClientExited == False:
            self.ClientX = 820
            self.ClientY = 50
            print("Client should exit")
            self.ClientExited = True

    def ClientWalkAnimation(self, screen):
        self.ClientImg1 = self.ClientImg[int(self.ClientImgIndex)]
        self.ClientImg1 = pygame.transform.scale(self.ClientImg1, (38, 72))
        screen.blit(self.ClientImg1, (self.ClientX, self.ClientY))

    def ClientShopExit(self, screen):
        if self.ClientX <= 1250:
            self.ClientWalkAnimation(screen)
            self.ClientImgIndex += 0.1
            if self.ClientImgIndex >= 3:
                self.ClientImgIndex = 0
            self.ClientX += self.ClientX_change
        self.ClientWalkAnimation(screen)


class Car(pygame.sprite.Sprite):
    def __init__(self,ParkingSpot,CarImgIndex,Clientnumberstr):
        super().__init__()
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
        self.CarImgIndex = CarImgIndex
        self.original_image =  pygame.transform.rotate(pygame.transform.scale(self.CarImg[int(self.CarImgIndex)], (70, 150)),-90)
        self.image = self.original_image
        self.rect =  self.image.get_rect(center = (-100,160))
        self.ParkingSpotfont = pygame.font.SysFont('Comic Sans MS', 20)
        self.ParkingSpot = ParkingSpot
        self.angle = 0
        self.rotation_speed = 1.5
        self.direction = 0
        self.active = False
        self.ClientEntered = False
        self.forward = pygame.math.Vector2(1,0)
        self.activeforward = False
        self.activebackward = False
        self.SuccessDelivery = False
        self.entrycar = False
        self.Client = pygame.sprite.GroupSingle()
        self.Client.add(Client(Clientnumberstr))

    def set_rotation(self):
        if self.direction == 1: 
            self.angle -= self.rotation_speed
        if self.direction == -1:
            self.angle += self.rotation_speed
        self.image = pygame.transform.rotate(self.original_image,self.angle)
        self.rect = self.image.get_rect(center = self.rect.center)

    def get_rotation(self):
        if self.direction == 1:
            self.forward.rotate_ip(self.rotation_speed)
        if self.direction == -1:
            self.forward.rotate_ip(-self.rotation_speed)

    def accelerate(self):
        if self.activeforward:
            self.rect.center += self.forward * 3
        if self.activebackward:
            self.rect.center -= self.forward * 3


    def boundaries(self):
        if self.rect.x <= 90:
            self.rect.x = 90
        elif self.rect.x >= 1130:
            self.rect.x = 1130
        if self.rect.y <= 125:
            self.rect.y = 125
        elif self.rect.y >= 610:
            self.rect.y = 610

    def initial_animation(self, screen):
        if self.rect.x <= 40:
            self.rect.x += 3
        if self.rect.x >= 40:
            self.Client.sprite.ClientX += self.Client.sprite.ClientX_change
            self.Client.sprite.ClientWalkAnimation(screen)
            self.Client.sprite.ClientImgIndex += 0.1
            if self.Client.sprite.ClientImgIndex >= 3:
                self.Client.sprite.ClientImgIndex =0
            
            #screen.draw.text(str(self.ParkingSpots), (self.ClientX, self.ClientY - 30), color=(200, 200, 200), background=WHITE)
            if self.Client.sprite.ClientX <= 300:
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
            print("Exit spot reached")
            self.rect.x = 1600
            self.Client.sprite.ClientX = 1500
            self.SuccessDelivery = True
    
    def ClientExit(self):
            if self.ClientEntered:
                self.Client.sprite.ClientExit()
    

    def update(self,screen):
        if self.ClientEntered != True:
            self.initial_animation(screen)
        if self.active:
            self.set_rotation()
            self.get_rotation()
            self.accelerate()
            self.boundaries()
        if self.Client.sprite.ClientExited:
            self.Client.sprite.ClientShopExit(screen)
            if self.active == False:
                self.successfulDelivery()
        if self.SuccessDelivery == True:
            self.kill()