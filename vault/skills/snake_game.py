
import pygame
from pygame import mixer
pygame.init()
mixer.init()
screen = pygame.display.set_mode((800,650)) # Set the width and height of your game window here
clock=pygame.time.Clock()#For setting FPS
score = 0 
snake_speed = 15  
font  = pygame.font.Font('freesansbold.ttf',30)   
white = (255,255,255)    
black=(0,0,0)