import pygame
from random import randint,choice, shuffle
from os import walk


ParkingSpot = [1001,1002,1003,1004,1005,1006,1007,1008,1009,1010,1011,1012,1013,1014,1015,1016,1017,1018]


def GameTimer(seconds,GameTime,GameTimeFont,screen,entrance_blocked):
    if entrance_blocked != 0:
        Gameseconds = GameTime - round(seconds)
        GameTimeRender = GameTimeFont.render(str(Gameseconds), True, (0, 0, 0))
        pygame.draw.rect(screen, (255,255,255), (1085, 5,75, 45))
        screen.blit(GameTimeRender,(1100, 10))
    if entrance_blocked:
        Gameseconds = GameTime - int((1.2*(round(seconds))))
        GameTimeRender = GameTimeFont.render(str(Gameseconds), True, (136, 8, 8))
        pygame.draw.rect(screen, (255,255,255), (1085, 5,75, 45))
        screen.blit(GameTimeRender,(1100, 10))
    if Gameseconds < 0:
        return True
    else:
        return False

def import_folder(path):
    surface_list = []
    for img_path,folders,files in walk(path):
        for img_file in files:
            full_path = path + '/' + img_file
            img_surface = pygame.image.load(full_path).convert_alpha()
            img_surface = pygame.transform.scale(img_surface, (38, 72))
            surface_list.append(img_surface)

    return surface_list


class Spots(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.Entrance_spot_img = pygame.image.load('Resources\Entrance_spot.png').convert_alpha()
        self.image = pygame.transform.scale(self.Entrance_spot_img, (250, 120))
        self.rect = self.image.get_rect(center = (0,177))

    


# def GenerateClientArrivalTime(GameTimeEnd, GameTimestart, numb_clients):
#     #allocate arrival times to the guests
#     GameTimeEndArrival = GameTimeEnd - 20 # no one should arrive after this time
#     GameTimeStartArrival = GameTimestart # no one should arrive before this
#     number_clients = numb_clients # number of clients to arrive in the game
#     ClientsArrivalTime = []
#     i = 0
#     while i  < (number_clients):
#         random_time = randint(GameTimeStartArrival,GameTimeEndArrival)
#         if random_time not in ClientsArrivalTime:
#             ClientsArrivalTime.append(random_time)
#             i += 1
#         else:
#             i -= 1
#     return ClientsArrivalTime

# def GenerateClientExitTime(ArrivalTime,GameTime,person_array):
#     GameTimeEndArrival = GameTime - 15 # no one should come to exit after this
#     ClientsExitTime = []
#     i = 0
#     while i < (len(ArrivalTime)):
#         random_time = randint(ArrivalTime[i],GameTimeEndArrival)
#         if random_time not in ClientsExitTime:
#             ClientsExitTime.append(random_time)
#         else:
#             if i == 0:
#                 i = 0
#             else:
#                 i -= 1
#         i += 1
#     return ClientsExitTime

        
def CarSelection(GameTime,Parking_spots,numb_clients,numb_cars):
    #this function is for car and client selection. car can be non unique, however avatar and parking number for client is always unique until he exits.
    m = numb_clients # number of ppl
    n = 3 # number of attributes
    GameTimeStart = 5 # no client should come before this
    ClientsList = [[0 for x in range(n)] for x in range(m)]
    person_number = ["person1","person2","person3","person4","person5","person6","person7","person8","person9","person10","person11","person12","person13","person14","person15","person16","person17"]
    shuffle(person_number) # shuffle for randomization of ppl coming in or exiting
    shuffle(Parking_spots)
    carcollection = numb_cars # Total number of cars available
    # ClientsArrivalTimes = GenerateClientArrivalTime(GameTime,GameTimeStart,numb_clients) # A list of times for the Clients to arrive
    # ClientsExitTimes = GenerateClientExitTime(ClientsArrivalTimes,GameTime,person_number)
    for i in range(m):
        ClientsList[i][0]= randint(0,carcollection)  # car assigned to the client
        ClientsList[i][1] = person_number.pop()
        ClientsList[i][2] = Parking_spots.pop()
        #ClientsList[i][3] = ClientsArrivalTimes.pop()
        #ClientsList[i][4] = ClientsExitTimes.pop()

        #print(str(ClientsList[i][0]) + " " + ClientsList[i][1] + " " + str(ClientsList[i][2]) +" " + str(ClientsList[i][3]) + " " + str(ClientsList[i][4]))
    return ClientsList

def CarCollisions(CarGroup,collisionpenalty):
    # this function looks out for car collisions and returns the object that collides and also induces the penalty term in the game run time
    collision = False
    #returns the car objects that collide and also the updated time

def Entrance_blocked():
    entrance_blocked = False
 # check entrance blocked
 # display entrance blocked 
 # if entrance blocked for more than 1 cars, adjust all the n - 1 arrival times

    return entrance_blocked