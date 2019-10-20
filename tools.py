import os
import pygame as pg


def cartesian2py(point):
    return point.x * 10 + 200, - point.y * 10


def loadImage(name):
    fullname = os.path.join("data", name)
    try:
        image = pg.image.load(fullname)
    except pg.error as message:
        print("Cannot load image", name)
        raise SystemExit(message)
    image = image.convert()
    return image


class Point:

    def __init__(self, x, y):
        self.x = x
        self.y = y


class Color:
    white = (255, 255, 255)
    black = (0, 0, 0)


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
