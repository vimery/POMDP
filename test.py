# import pygame
# import sys
# from tools import loadImage
#
# pygame.init()
#
# win = pygame.display.set_mode((300, 300))
# caption = pygame.display.set_caption("hello")
# image = loadImage('vehicle.png')
# image_rect = image.get_rect()
# image_rect = image_rect.move(100, 100)
# angle = 1
# v = 2
# while True:
#     win.fill((255, 255, 255))
#     newImage = pygame.transform.rotate(image, angle)
#     new_rect = newImage.get_rect(center=image_rect.center)
#     angle += 2
#     win.blit(newImage, new_rect)
#     pygame.display.update()
#     pygame.time.delay(100)
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             sys.exit(0)
import pygame as pg
from roadmap import *
from tools import *
pg.init()
win = pg.display.set_mode((400, 400))
pg.display.set_caption("Autonomous driving simulator")
background = win.fill(Color.white, (0, 0, 400, 400))
font = pg.font.SysFont("arial", 16)

params = InterParam()
x_min = params.x_min
x_max = params.x_max
y_min = params.y_min
# position of intersection, default is (0,0)
inter_x = params.inter_x
inter_y = params.inter_y
# shape of intersection, default is square
inter_width = params.inter_width
inter_height = params.inter_height
# shape of line, default is 1/2 * inter_width
line_width = params.line_width
# speed limit
max_speed = params.max_speed

seg1 = Segment(x_max, -line_width / 2, x_max - inter_x - inter_width / 2, line_width, math.pi, max_speed)
seg1.render(win)
pg.display.update()
pg.time.wait(5000)

