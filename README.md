# Car-QLearning

Requirements (versions are mostly here as an indication):

```
Pyglet (1.5.27)
Pygame (2.1.3)
numpy (1.24.1)
tensorflow (2.10.1) (no 1.X tensorflow)
```

How to use : 

I) Download the files 


II) If you want to create your own track


1) Create your own .png file (I recommend not to change size if you don't want to touch the code).
When you are done name it track.png. 

3) Once done designing, in Games_Solo.py empty the set_wall function and the set-gates function but leave this line in set_gates :
  self.gates.append(RewardGate(0, 1, 2, 3))

4) You can now run Main_Solo.py program, you should see you track

5) Time to set up the gates. You can setup gates using your mouse left click. These gate are where the AI gain points. Once you are done close the program you should see in the console a lot of text similar to :
```
self.gates.append(RewardGate(343, 379, 524, 405))
self.gates.append(RewardGate(488, 326, 626, 421))
self.gates.append(RewardGate(626, 309, 701, 411))
self.gates.append(RewardGate(232, 309, 267, 399))
```
Copy this text and paste it in the set_gates function you emptied earlier. Be careful the order of the gates is important.

6) Time to set up the Walls. In the Main_Solo.py program go to the "on_mouse_press" function then swap the lines in comments and the one that are not. You should get:

```
    def on_mouse_press(self, x, y, button, modifiers):
        if self.firstClick:
            self.clickPos = [x, y]
        else:
            #print("self.gates.append(RewardGate({}, {}, {}, {}))".format(self.clickPos[0],displayHeight - self.clickPos[1],x, displayHeight - y))
            #self.game.gates.append(RewardGate(self.clickPos[0], displayHeight - self.clickPos[1], x, displayHeight - y))

            print("self.walls.append(Wall({}, {}, {}, {}))".format(self.clickPos[0],displayHeight - self.clickPos[1],x, displayHeight - y))
            self.game.walls.append(Wall(self.clickPos[0], self.clickPos[1], x, y))

        self.firstClick = not self.firstClick
        pass
        
```
You can now setup walls using your mouse left click. Once you are done close the program you should see in the console a lot of text similar to before but saying :

```
self.walls.append(Wall(343, 379, 524, 405)) ...
```

Once again copy this text and paste it in the set_walls function of Games_Solo.py.

7) Change the start and reset position/direction of the car in Games_Solo.py in the car class in the __init__ function and in the reset function (change self.x, self.y)

8) Check everything is good by trying your track in the Main_Solo program you should die if you touch the walls and you should see gates disappear when passing them (if not you probably didn't placed the in the right order)

9) copy your set_walls and set_gates function into the Games.py by replacing the old one. (don't forget to change your start and reset position/direction too)

10) You are now done creating your personal track.


III) You now want to train the AI to perform on your track so run the Main.py program. You should see the the car moving on it's own and learning slowly the track.

PS : It seems that the load or save fuction aren't working properly so don't close your program until your are satisfied =) (I personally got result at around Model 5000 so be patient)

If you need some help you can contact me by discord at Aquinox#4429. I'll try to help you as best as I can.
