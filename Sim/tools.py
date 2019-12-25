import os
import pygame as pg
from pygame.locals import RLEACCEL
import math
import numpy as np


class Color:
    white = (255, 255, 255)
    black = (0, 0, 0)
    green = (0, 255, 0)


def cartesian2py(x, y):
    scale = InterParam.scale
    return round(x * scale + InterParam.x_max * scale), round(- y * scale + InterParam.y_max * scale + scale)


def load_image(name, width, height, color_key=Color.white):
    fullname = os.path.join("data", name)
    try:
        image = pg.image.load(fullname)
    except pg.error as message:
        print("Cannot load image", name)
        raise SystemExit(message)
    # time 2 to let the image larger
    image = pg.transform.scale(image, (round(height * 2 * InterParam.scale), round(width * 2 * InterParam.scale)))
    image = image.convert_alpha()
    image.set_colorkey(color_key, RLEACCEL)
    return image


def collide_detection(x1, y1, x2, y2, r1, r2):
    x_collide = x1 + r1 >= x2 and x2 + r2 >= x1
    y_collide = y1 + r1 >= y2 and y2 + r2 >= y1
    # return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) <= r1 + r2
    return x_collide and y_collide


def forward(x, y, theta, v, route, dt, max_v, a):
    if v + a * dt > max_v:
        acc_t = (max_v - v) / a
        distance = (max_v + v) / 2 * acc_t + max_v * (dt - acc_t)
    else:
        distance = v * dt + a * dt * dt
    return route.next(x, y, theta, distance)


def draw_dashed_line(surf, color, start_pos, end_pos, width=1, dash_length=5):
    x1, y1 = start_pos
    x2, y2 = end_pos
    dl = dash_length

    if x1 == x2:
        y_coords = [y for y in range(y1, y2, dl if y1 < y2 else -dl)]
        x_coords = [x1] * len(y_coords)
    elif y1 == y2:
        x_coords = [x for x in range(x1, x2, dl if x1 < x2 else -dl)]
        y_coords = [y1] * len(x_coords)
    else:
        a = abs(x2 - x1)
        b = abs(y2 - y1)
        c = round(math.sqrt(a ** 2 + b ** 2))
        dx = dl * a / c
        dy = dl * b / c

        x_coords = [x for x in np.arange(x1, x2, dx if x1 < x2 else -dx)]
        y_coords = [y for y in np.arange(y1, y2, dy if y1 < y2 else -dy)]

    next_coords = list(zip(x_coords[1::2], y_coords[1::2]))
    last_coords = list(zip(x_coords[0::2], y_coords[0::2]))
    for (x1, y1), (x2, y2) in zip(next_coords, last_coords):
        start = (round(x1), round(y1))
        end = (round(x2), round(y2))
        pg.draw.line(surf, color, start, end, width)


class InterParam:
    """
    InterParam: parameters for constructing an intersection
    """
    x_min = -15
    x_max = 15
    y_min = -15
    y_max = 15
    inter_x = 0.0
    inter_y = 0.0
    inter_width = 10.0
    num_lanes = 2
    scale = 10

    max_speed = 4.0  # m/s
