import pygame


def DrawParkingSpots(numSpots,screen,parkingfont):
    depthx= 100
    heighty= 150
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
        if i <= 5:
            drawtopspot(initialx,initialy,depthx,heighty,spotnumber,screen,parkingfont)
            spotnumber += 1
            initialx += depthx
        if i > 5 and i<= 11:
            drawbotspot(botinitialx,2*initialy,depthx,heighty,spotnumber,screen,parkingfont)
            spotnumber += 1
            botinitialx += depthx
        if i > 11 and i <= 14:
            drawleftspot(leftspotx,botlefty,depthx,heighty,spotnumber,screen,parkingfont)
            spotnumber += 1
            botlefty += depthx
        if i > 14 and i <= 17:
            drawrightspot(rightspotx,botrighty,depthx,heighty,spotnumber,screen,parkingfont)
            spotnumber += 1
            botrighty += depthx


def drawtopspot(initialx,initialy,depthx,heighty,spotnumber,screen,parkingfont):
    pygame.draw.lines(screen,( 	242, 241, 221), False, [[initialx,initialy],[initialx,initialy + heighty],[initialx+depthx, initialy+heighty],[initialx+depthx,initialy]],5)
    spotnumber = parkingfont.render(str(spotnumber), True, ( 	255, 253, 208))
    screen.blit(spotnumber, (initialx + 25, initialy + 120))

def drawbotspot(initialx,initialy,depthx,heighty,spotnumber,screen,parkingfont):
    pygame.draw.lines(screen, ( 	242, 241, 221), False, [[initialx,initialy],[initialx,initialy - heighty],[initialx+depthx, initialy-heighty],[initialx+depthx,initialy]],5)
    spotnumber = parkingfont.render(str(spotnumber), True, ( 255, 253, 208))
    screen.blit(spotnumber, (initialx + 25, initialy - 120))

def drawleftspot(initialx,initialy,depthx,heighty,spotnumber,screen,parkingfont):
    pygame.draw.lines(screen, (	242, 241, 221), False, [[initialx,initialy],[initialx-heighty,initialy],[initialx-heighty, initialy+depthx],[initialx,initialy+depthx]],5)
    spotnumber = parkingfont.render(str(spotnumber), True, ( 255, 253, 208))
    spotnumber = pygame.transform.rotate(spotnumber, 90)
    screen.blit(spotnumber, (initialx - heighty +10, initialy +30 ))

def drawrightspot(initialx,initialy,depthx,heighty,spotnumber,screen,parkingfont):
    pygame.draw.lines(screen, (	242, 241, 221), False, [[initialx,initialy],[initialx+heighty,initialy],[initialx+heighty, initialy+depthx],[initialx,initialy+depthx]],5)
    spotnumber = parkingfont.render(str(spotnumber), True, ( 255, 253, 208))
    spotnumber = pygame.transform.rotate(spotnumber, 90)
    screen.blit(spotnumber, (initialx +heighty -30 , initialy +30 ))

