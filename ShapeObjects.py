import pyglet
from pyglet.gl import *
import math




class Triangle:
    def __init__(self, x1, y1, x2, y2, x3, y3, col=[255, 255, 255]):
        self.vertices = pyglet.graphics.vertex_list(3, ('v3f', [x1, y1, 0, x2, y2, 0, x3, y3, 0]),
                                                    ('c3B', [*col, *col, *col]))

    def show(self):
        self.vertices.draw(GL_TRIANGLES)


class Rect:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.col = [255, 0, 0] * 4
        # self.vertices = pyglet.graphics.vertex_list('v3f')

    def setColor(self, newColor):
        self.col = newColor * 4

    def draw(self):
        x = self.x
        y = self.y
        w = self.w
        h = self.h
        pyglet.graphics.draw_indexed(4, pyglet.gl.GL_TRIANGLES, [0, 1, 2, 0, 2, 3],
                                     ('v2f', [x, y, x + w, y, x + w, y + h, x, y + h]),
                                     ('c3B', self.col))


class Line:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.color = [0, 0, 0] * 2
        self.lineThinkness = 1

    def draw(self):
        pyglet.gl.glLineWidth(self.lineThinkness)
        pyglet.graphics.draw(2, pyglet.gl.GL_LINES,
                             ("v2f", (self.x1, self.y1, self.x2, self.y2))
                             , ('c3B', self.color))

    def setColor(self, newColor):
        self.color = newColor * 2

    def setLineThinkness(self, thinkness):
        self.lineThinkness = thinkness
