import pyglet
from pyglet.window import key
from pyglet.gl import *
from Global import *
import pygame
from Games_Solo import Game, Wall, RewardGate

vec2 = pygame.math.Vector2

class MyWindow(pyglet.window.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_minimum_size(400, 300)
        # set background color
        backgroundColor = [0, 0, 0, 255]
        backgroundColor = [i / 255 for i in backgroundColor]
        glClearColor(*backgroundColor)
        self.clear()
        self.game = Game()


        self.firstClick = True

    def on_key_press(self, symbol, modifiers):
        if symbol == key.RIGHT:
            self.game.car.turningRight = True

        if symbol == key.LEFT:
            self.game.car.turningLeft = True

        if symbol == key.UP:
            self.game.car.accelerating = True

        if symbol == key.DOWN:
            self.game.car.reversing = True

    def on_key_release(self, symbol, modifiers):
        if symbol == key.RIGHT:
            self.game.car.turningRight = False

        if symbol == key.LEFT:
            self.game.car.turningLeft = False

        if symbol == key.UP:
            self.game.car.accelerating = False

        if symbol == key.DOWN:
            self.game.car.reversing = False

        if symbol == key.SPACE:
            self.ai.training = not self.ai.training

    def on_mouse_press(self, x, y, button, modifiers):
        if self.firstClick:
            self.clickPos = [x, y]
        else:
            print("self.gates.append(RewardGate({}, {}, {}, {}))".format(self.clickPos[0],displayHeight - self.clickPos[1],x, displayHeight - y))
            self.game.gates.append(RewardGate(self.clickPos[0], displayHeight - self.clickPos[1], x, displayHeight - y))

            #print("self.walls.append(Wall({}, {}, {}, {}))".format(self.clickPos[0],displayHeight - self.clickPos[1],x, displayHeight - y))
            #self.game.walls.append(Wall(self.clickPos[0], self.clickPos[1], x, y))

        self.firstClick = not self.firstClick
        pass


    def on_draw(self):

        window.set_size(width=displayWidth,height=displayHeight)
        self.clear()
        self.game.render()
        vision = self.game.car.getState()
        for i in range(len(vision)):

            label = pyglet.text.Label("{}:  {}".format(i,vision[i]),
                                       font_name='Times New Roman',
                                       font_size=12,
                                       x=10, y=20*i+50,
                                       anchor_x='left', anchor_y='center')
            label.draw()


    def update(self,dt):
        pass



if __name__ == "__main__":
    window = MyWindow(displayWidth, displayHeight, "AI Learns to Drive", resizable=False)
    pyglet.clock.schedule_interval(window.update, 1 / frameRate)
    pyglet.app.run()