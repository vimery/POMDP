import pygame
import sys
pygame.init()

win = pygame.display.set_mode((300, 300))
caption = pygame.display.set_caption("hello")
win.fill((0, 0, 0))
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit(0)
