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
# import pygame as pg
# from roadmap import *
# from tools import *
#
# pg.init()
# params = InterParam()
# win = pg.display.set_mode((params.x_max * 20, params.y_max * 20))
# pg.display.set_caption("Autonomous driving simulator")
# background = win.fill(Color.white, (0, 0, params.x_max * 20, params.y_max * 20))
# font = pg.font.SysFont("arial", 16)

# line_width = params.inter_width / 2
# # horizon
# r_start_x = params.x_max
# r_length = params.x_max - line_width
# l_start_x = params.x_min
# l_length = -params.x_min - line_width
# # vertical
# d_start_y = params.y_min
# d_length = -params.y_min - line_width
# t_start_y = params.y_max
# t_length = params.y_max - line_width
#
# p = line_width / 2
# seg1 = Segment(r_start_x, p, r_length, line_width, pi)
# seg2 = Segment(-line_width, p, l_length, line_width, pi)
# seg3 = Segment(l_start_x, -p, l_length, line_width, 0)
# seg4 = Segment(line_width, -p, r_length, line_width, 0)
# seg5 = Segment(-p, -line_width, d_length, line_width, nh_pi)
# seg6 = Segment(p, d_start_y, d_length, line_width, h_pi)
# seg7 = Segment(-p, t_start_y, t_length, line_width, nh_pi)
# seg8 = Segment(p, line_width, t_length, line_width, h_pi)

# seg1.render(win)
# seg2.render(win)
# seg3.render(win)
# seg4.render(win)
# seg5.render(win)
# seg6.render(win)
# seg7.render(win)
# seg8.render(win)

# pg.display.update()
# pg.time.wait(5000)
