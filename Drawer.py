import pyglet
from pyglet.gl import *
import math

class Drawer:

    def __init__(self):
        self.color = [100, 0, 0]
        self.lineThickness = 1

    def setLineThinkness(self, thinkness):
        self.lineThickness = thinkness

    def setColor(self, newColor):
        self.color = newColor

    def line(self, x1, y1, x2, y2):
        pyglet.graphics.draw(2, pyglet.gl.GL_LINES,
                             ("v2f", (x1, y1, x2, y2))
                             , ('c3B', self.color * 2))

    def rect(self, x, y, w, h):
        pyglet.graphics.draw_indexed(4, pyglet.gl.GL_TRIANGLES, [0, 1, 2, 0, 2, 3],
                                     ('v2f', [x, y, x + w, y, x + w, y + h, x, y + h]),
                                     ('c3B', self.color * 4))

    def triangle(self, x1, y1, x2, y2, x3, y3):
        pyglet.graphics.draw(3, pyglet.gl.GL_TRIANGLES,
                             ('v2f', [x1, y1, x2, y2, x3, y3]),
                             ('c3B', self.color * 3))

    def circle(self, x, y, radius):
        iterations = int(2 * radius * math.pi)
        s = math.sin(2 * math.pi / iterations)
        c = math.cos(2 * math.pi / iterations)

        dx, dy = radius, 0

        glBegin(GL_TRIANGLE_FAN)
        gl.glColor4f(self.color[0] / 255, self.color[1] / 255, self.color[2] / 255, 1.0)
        glVertex2f(x, y)
        for i in range(iterations + 1):
            glVertex2f(x + dx, y + dy)
            dx, dy = (dx * c - dy * s), (dy * c + dx * s)
        glEnd()
