import os
import pygame as pg
from pygame.locals import RLEACCEL
import math
import numpy as np


class Color:
    white = (255, 255, 255)
    black = (0, 0, 0)


def cartesian2py(x, y):
    return round(x * 10 + 200), round(- y * 10 + 10)


def loadImage(name, colorkey=Color.white):
    fullname = os.path.join("data", name)
    try:
        image = pg.image.load(fullname)
    except pg.error as message:
        print("Cannot load image", name)
        raise SystemExit(message)
    image = image.convert()
    image.set_colorkey(colorkey, RLEACCEL)
    return image


class Point:

    def __init__(self, x, y):
        self.x = x
        self.y = y


def draw_dashed_line(surf, color, start_pos, end_pos, width=1, dash_length=5):
    x1, y1 = start_pos
    x2, y2 = end_pos
    dl = dash_length

    if x1 == x2:
        ycoords = [y for y in range(y1, y2, dl if y1 < y2 else -dl)]
        xcoords = [x1] * len(ycoords)
    elif y1 == y2:
        xcoords = [x for x in range(x1, x2, dl if x1 < x2 else -dl)]
        ycoords = [y1] * len(xcoords)
    else:
        a = abs(x2 - x1)
        b = abs(y2 - y1)
        c = round(math.sqrt(a ** 2 + b ** 2))
        dx = dl * a / c
        dy = dl * b / c

        xcoords = [x for x in np.arange(x1, x2, dx if x1 < x2 else -dx)]
        ycoords = [y for y in np.arange(y1, y2, dy if y1 < y2 else -dy)]

    next_coords = list(zip(xcoords[1::2], ycoords[1::2]))
    last_coords = list(zip(xcoords[0::2], ycoords[0::2]))
    for (x1, y1), (x2, y2) in zip(next_coords, last_coords):
        start = (round(x1), round(y1))
        end = (round(x2), round(y2))
        pg.draw.line(surf, color, start, end, width)


class InterParam:
    """
    InterParam: parameters for constructing an intersection
    """
    x_min = -20
    x_max = 20
    y_min = -20
    inter_x = 0.0
    inter_y = 0.0
    inter_width = 6.0
    inter_height = 6.0
    line_width = inter_width / 2
    num_lanes = 2

    max_speed = 3.0  # m/s
