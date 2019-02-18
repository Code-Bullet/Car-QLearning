import math
import pygame

vec2 = pygame.math.Vector2


def get_angle(vec):
    if vec.length() == 0:
        return 0
    return math.degrees(math.atan2(vec.y, vec.x))


def angleToRadians(angle):
    return angle / (180 / math.pi)


def radiansToAngle(rads):
    return rads * 180 / math.pi


def linesCollided(x1, y1, x2, y2, x3, y3, x4, y4):
    uA = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / ((y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1))
    uB = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / ((y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1))
    if 0 <= uA <= 1 and 0 <= uB <= 1:
        return True
    return False


def getCollisionPoint(x1, y1, x2, y2, x3, y3, x4, y4):
    global vec2
    uA = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / ((y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1))
    uB = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / ((y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1))
    if 0 <= uA <= 1 and 0 <= uB <= 1:
        intersectionX = x1 + (uA * (x2 - x1))
        intersectionY = y1 + (uA * (y2 - y1))
        return vec2(intersectionX, intersectionY)
    return None


def dist(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
