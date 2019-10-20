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

a = [1, 2, 3, 4, 5]
del a[2]
