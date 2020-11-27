import numpy as np
import pyglet
from Globals import displayWidth, displayHeight
from Drawer import Drawer
from ShapeObjects import *
from PygameAdditionalMethods import *
import pygame

drawer = Drawer()
vec2 = pygame.math.Vector2


class Game:
    no_of_actions = 9
    state_size = 15

    def __init__(self):
        trackImg = pyglet.image.load('images/track.png')
        self.trackSprite = pyglet.sprite.Sprite(trackImg, x=0, y=0)
        # initiate car

        # initiate walls
        self.walls = []
        self.gates = []

        self.set_walls()
        self.set_gates()
        self.firstClick = True

        self.car = Car(self.walls, self.gates)

    def set_walls(self):
        self.walls.append(Wall(240, 809, 200, 583))
        self.walls.append(Wall(200, 583, 218, 395))
        self.walls.append(Wall(218, 395, 303, 255))
        self.walls.append(Wall(303, 255, 548, 173))
        self.walls.append(Wall(548, 173, 764, 179))
        self.walls.append(Wall(764, 179, 1058, 198))
        self.walls.append(Wall(1055, 199, 1180, 215))
        self.walls.append(Wall(1177, 215, 1220, 272))
        self.walls.append(Wall(1222, 273, 1218, 367))
        self.walls.append(Wall(1218, 367, 1150, 437))
        self.walls.append(Wall(1150, 437, 1044, 460))
        self.walls.append(Wall(1044, 460, 757, 600))
        self.walls.append(Wall(757, 600, 1099, 570))
        self.walls.append(Wall(1100, 570, 1187, 508))
        self.walls.append(Wall(1187, 507, 1288, 443))
        self.walls.append(Wall(1288, 443, 1463, 415))
        self.walls.append(Wall(1463, 415, 1615, 478))
        self.walls.append(Wall(1617, 479, 1727, 679))
        self.walls.append(Wall(1727, 679, 1697, 874))
        self.walls.append(Wall(1694, 872, 1520, 964))
        self.walls.append(Wall(1520, 964, 1100, 970))
        self.walls.append(Wall(1105, 970, 335, 960))
        self.walls.append(Wall(339, 960, 264, 899))
        self.walls.append(Wall(263, 897, 238, 803))
        self.walls.append(Wall(317, 782, 274, 570))
        self.walls.append(Wall(275, 569, 284, 407))
        self.walls.append(Wall(284, 407, 363, 317))
        self.walls.append(Wall(363, 317, 562, 240))
        self.walls.append(Wall(562, 240, 1114, 284))
        self.walls.append(Wall(1114, 284, 1120, 323))
        self.walls.append(Wall(1120, 323, 1045, 377))
        self.walls.append(Wall(1045, 378, 682, 548))
        self.walls.append(Wall(682, 548, 604, 610))
        self.walls.append(Wall(604, 612, 603, 695))
        self.walls.append(Wall(605, 695, 702, 713))
        self.walls.append(Wall(703, 712, 1128, 642))
        self.walls.append(Wall(1129, 642, 1320, 512))
        self.walls.append(Wall(1323, 512, 1464, 497))
        self.walls.append(Wall(1464, 497, 1579, 535))
        self.walls.append(Wall(1579, 535, 1660, 701))
        self.walls.append(Wall(1660, 697, 1634, 818))
        self.walls.append(Wall(1634, 818, 1499, 889))
        self.walls.append(Wall(1499, 889, 395, 883))
        self.walls.append(Wall(395, 883, 330, 838))
        self.walls.append(Wall(330, 838, 315, 782))
        self.walls.append(Wall(319, 798, 306, 725))
        self.walls.append(Wall(276, 580, 277, 543))
        self.walls.append(Wall(603, 639, 622, 590))
        self.walls.append(Wall(599, 655, 621, 704))
        self.walls.append(Wall(1074, 571, 1115, 558))
        self.walls.append(Wall(1314, 516, 1333, 511))
        self.walls.append(Wall(1692, 875, 1706, 830))
        self.walls.append(Wall(277, 912, 255, 872))
        self.walls.append(Wall(1214, 262, 1225, 288))
        self.walls.append(Wall(1601, 470, 1625, 490))
        self.walls.append(Wall(1119, 644, 1139, 634))
        self.walls.append(Wall(687, 710, 719, 710))
        self.walls.append(Wall(1721, 664, 1727, 696))
        self.walls.append(Wall(1015, 392, 1065, 362))
        self.walls.append(Wall(1091, 572, 1104, 568))
        self.walls.append(Wall(1157, 528, 1233, 478))

    def set_gates(self):
        self.gates.append(RewardGate(314, 345, 200, 326))
        self.gates.append(RewardGate(187, 435, 311, 451))
        self.gates.append(RewardGate(307, 537, 171, 555))
        self.gates.append(RewardGate(234, 681, 345, 628))
        self.gates.append(RewardGate(408, 682, 363, 788))
        self.gates.append(RewardGate(428, 816, 481, 712))
        self.gates.append(RewardGate(568, 733, 543, 854))
        self.gates.append(RewardGate(678, 858, 675, 710))
        self.gates.append(RewardGate(852, 708, 855, 848))
        self.gates.append(RewardGate(995, 836, 985, 705))
        self.gates.append(RewardGate(1059, 710, 1076, 821))
        self.gates.append(RewardGate(1078, 667, 1172, 572))
        self.gates.append(RewardGate(997, 616, 1076, 532))
        self.gates.append(RewardGate(967, 492, 909, 566))
        self.gates.append(RewardGate(788, 512, 839, 438))
        self.gates.append(RewardGate(790, 405, 781, 285))
        self.gates.append(RewardGate(891, 302, 899, 427))
        self.gates.append(RewardGate(1004, 434, 1027, 334))
        self.gates.append(RewardGate(1139, 344, 1084, 452))
        self.gates.append(RewardGate(1171, 502, 1233, 416))
        self.gates.append(RewardGate(1305, 454, 1243, 556))
        self.gates.append(RewardGate(1365, 588, 1408, 480))
        self.gates.append(RewardGate(1487, 472, 1524, 587))
        self.gates.append(RewardGate(1642, 508, 1575, 432))
        self.gates.append(RewardGate(1608, 360, 1709, 419))
        self.gates.append(RewardGate(1744, 324, 1625, 296))
        self.gates.append(RewardGate(1609, 231, 1727, 190))
        self.gates.append(RewardGate(1617, 66, 1541, 163))
        self.gates.append(RewardGate(1487, 135, 1510, 14))
        self.gates.append(RewardGate(1344, 16, 1328, 150))
        self.gates.append(RewardGate(1077, 142, 1067, 14))
        self.gates.append(RewardGate(909, 16, 900, 130))
        self.gates.append(RewardGate(718, 138, 698, 20))
        self.gates.append(RewardGate(551, 18, 567, 132))
        self.gates.append(RewardGate(445, 138, 413, 13))
        self.gates.append(RewardGate(379, 154, 243, 80))
        self.gates.append(RewardGate(357, 221, 203, 182))

    def new_episode(self):
        self.car.reset()


    def get_state(self):
        return self.car.getState()
        pass

    def make_action(self, action):
        # returns reward
        actionNo = np.argmax(action)
        self.car.updateWithAction(actionNo)
        return self.car.reward

    def is_episode_finished(self):
        return self.car.dead

    def get_score(self):
        return self.car.score

    def get_lifespan(self):
        return self.car.lifespan

    def render(self):
        glPushMatrix()
        #
        # glTranslatef(-1, -1, 0)
        # glScalef(1 / (displayWidth / 2), 1 / (displayHeight / 2), 1)

        # self.clear()
        self.trackSprite.draw()
        self.car.show()

        # for w in self.walls:
        #     w.draw()
        # for g in self.gates:
        #     g.draw()

        glPopMatrix()


class Wall:

    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = displayHeight - y1
        self.x2 = x2
        self.y2 = displayHeight - y2

        self.line = Line(self.x1, self.y1, self.x2, self.y2)
        self.line.setLineThinkness(2)

    """
    draw the line
    """

    def draw(self):
        self.line.draw()

    """
    returns true if the car object has hit this wall
    """

    def hitCar(self, car):
        global vec2
        cw = car.width
        # since the car sprite isn't perfectly square the hitbox is a little smaller than the width of the car
        ch = car.height - 4
        rightVector = vec2(car.direction)
        upVector = vec2(car.direction).rotate(-90)
        carCorners = []
        cornerMultipliers = [[1, 1], [1, -1], [-1, -1], [-1, 1]]
        carPos = vec2(car.x, car.y)
        for i in range(4):
            carCorners.append(carPos + (rightVector * cw / 2 * cornerMultipliers[i][0]) +
                              (upVector * ch / 2 * cornerMultipliers[i][1]))

        for i in range(4):
            j = i + 1
            j = j % 4
            if linesCollided(self.x1, self.y1, self.x2, self.y2, carCorners[i].x, carCorners[i].y, carCorners[j].x,
                             carCorners[j].y):
                return True
        return False


"""
class containing all the game logic for moving and displaying the car
"""


class RewardGate:

    def __init__(self, x1, y1, x2, y2):
        global vec2
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.active = True

        self.line = Line(self.x1, self.y1, self.x2, self.y2)
        self.line.setLineThinkness(1)
        self.line.setColor([0, 255, 0])

        self.center = vec2((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    """
    draw the line
    """

    def draw(self):
        if self.active:
            self.line.draw()

    """
    returns true if the car object has hit this wall
    """

    def hitCar(self, car):
        if not self.active:
            return False

        global vec2
        cw = car.width
        # since the car sprite isn't perfectly square the hitbox is a little smaller than the width of the car
        ch = car.height - 4
        rightVector = vec2(car.direction)
        upVector = vec2(car.direction).rotate(-90)
        carCorners = []
        cornerMultipliers = [[1, 1], [1, -1], [-1, -1], [-1, 1]]
        carPos = vec2(car.x, car.y)
        for i in range(4):
            carCorners.append(carPos + (rightVector * cw / 2 * cornerMultipliers[i][0]) +
                              (upVector * ch / 2 * cornerMultipliers[i][1]))

        for i in range(4):
            j = i + 1
            j = j % 4
            if linesCollided(self.x1, self.y1, self.x2, self.y2, carCorners[i].x, carCorners[i].y, carCorners[j].x,
                             carCorners[j].y):
                return True
        return False


class Car:

    def __init__(self, walls, rewardGates):
        global vec2
        self.x = 258
        self.y = 288
        self.vel = 0
        self.direction = vec2(0, 1)
        self.direction = self.direction.rotate(180 / 12)
        self.acc = 0
        self.width = 40
        self.height = 20
        self.turningRate = 5.0 / self.width
        self.friction = 0.98
        self.maxSpeed = self.width / 4.0
        self.maxReverseSpeed = -1 * self.maxSpeed / 2.0
        self.accelerationSpeed = self.width / 160.0
        self.dead = False
        self.driftMomentum = 0
        self.driftFriction = 0.87
        self.lineCollisionPoints = []
        self.collisionLineDistances = []
        self.vectorLength = 300

        self.carPic = pyglet.image.load('images/car.png')
        self.carSprite = pyglet.sprite.Sprite(self.carPic, x=self.x, y=self.y)
        self.carSprite.update(rotation=0, scale_x=self.width / self.carSprite.width,
                              scale_y=self.height / self.carSprite.height)

        self.turningLeft = False
        self.turningRight = False
        self.accelerating = False
        self.reversing = False
        self.walls = walls
        self.rewardGates = rewardGates
        self.rewardNo = 0

        self.directionToRewardGate = self.rewardGates[self.rewardNo].center - vec2(self.x, self.y)

        self.reward = 0

        self.score = 0
        self.lifespan = 0
    """
    draws the car to the screen
    """

    def reset(self):
        global vec2
        self.x = 258
        self.y = 288
        self.vel = 0
        self.direction = vec2(0, 1)
        self.direction = self.direction.rotate(180 / 12)
        self.acc = 0
        self.dead = False
        self.driftMomentum = 0
        self.lineCollisionPoints = []
        self.collisionLineDistances = []

        self.turningLeft = False
        self.turningRight = False
        self.accelerating = False
        self.reversing = False
        self.rewardNo = 0
        self.reward = 0

        self.lifespan = 0
        self.score = 0
        for g in self.rewardGates:
            g.active = True

    def show(self):
        # first calculate the center of the car in order to allow the
        # rotation of the car to be anchored around the center
        upVector = self.direction.rotate(90)
        drawX = self.direction.x * self.width / 2 + upVector.x * self.height / 2
        drawY = self.direction.y * self.width / 2 + upVector.y * self.height / 2
        self.carSprite.update(x=self.x - drawX, y=self.y - drawY, rotation=-get_angle(self.direction))
        self.carSprite.draw()
        # self.showCollisionVectors()

    """
     returns a vector of where a point on the car is after rotation 
     takes the position desired relative to the center of the car when the car is facing to the right
    """

    def getPositionOnCarRelativeToCenter(self, right, up):
        global vec2
        w = self.width
        h = self.height
        rightVector = vec2(self.direction)
        rightVector.normalize()
        upVector = self.direction.rotate(90)
        upVector.normalize()

        return vec2(self.x, self.y) + ((rightVector * right) + (upVector * up))

    def updateWithAction(self, actionNo):
        self.turningLeft = False
        self.turningRight = False
        self.accelerating = False
        self.reversing = False

        if actionNo == 0:
            self.turningLeft = True
        elif actionNo == 1:
            self.turningRight = True
        elif actionNo == 2:
            self.accelerating = True
        elif actionNo == 3:
            self.reversing = True
        elif actionNo == 4:
            self.accelerating = True
            self.turningLeft = True
        elif actionNo == 5:
            self.accelerating = True
            self.turningRight = True
        elif actionNo == 6:
            self.reversing = True
            self.turningLeft = True
        elif actionNo == 7:
            self.reversing = True
            self.turningRight = True
        elif actionNo == 8:
            pass
        totalReward = 0

        for i in range(1):
            if not self.dead:
                self.lifespan+=1
                self.move()
                self.updateControls()

                if self.hitAWall():
                    self.dead = True
                    # return
                self.checkRewardGates()
                totalReward += self.reward

        self.setVisionVectors()

        # self.update()

        self.reward = totalReward

    """
    called every frame
    """

    def update(self):
        if not self.dead:
            self.updateControls()
            self.move()

            if self.hitAWall():
                self.dead = True
                # return
            self.checkRewardGates()
            self.setVisionVectors()

    def checkRewardGates(self):
        global vec2
        self.reward = -1
        if self.rewardGates[self.rewardNo].hitCar(self):
            self.rewardGates[self.rewardNo].active = False
            self.rewardNo += 1
            self.score += 1
            self.reward = 10
            if self.rewardNo == len(self.rewardGates):
                self.rewardNo = 0
                for g in self.rewardGates:
                    g.active = True
        self.directionToRewardGate = self.rewardGates[self.rewardNo].center - vec2(self.x, self.y)

    """
    changes the position of the car to account for acceleration, velocity, friction and drift
    """

    def move(self):
        global vec2
        self.vel += self.acc
        self.vel *= self.friction
        self.constrainVel()

        driftVector = vec2(self.direction)
        driftVector = driftVector.rotate(90)

        addVector = vec2(0, 0)
        addVector.x += self.vel * self.direction.x
        addVector.x += self.driftMomentum * driftVector.x
        addVector.y += self.vel * self.direction.y
        addVector.y += self.driftMomentum * driftVector.y
        self.driftMomentum *= self.driftFriction

        if addVector.length() != 0:
            addVector.normalize()

        addVector.x * abs(self.vel)
        addVector.y * abs(self.vel)

        self.x += addVector.x
        self.y += addVector.y

    """
    keeps the velocity of the car within the maximum and minimum speeds
    """

    def constrainVel(self):
        if self.maxSpeed < self.vel:
            self.vel = self.maxSpeed
        elif self.vel < self.maxReverseSpeed:
            self.vel = self.maxReverseSpeed

    """
    changes the cars direction and acceleration based on the users inputs
    """

    def updateControls(self):
        multiplier = 1
        if abs(self.vel) < 5:
            multiplier = abs(self.vel) / 5
        if self.vel < 0:
            multiplier *= -1

        driftAmount = self.vel * self.turningRate * self.width / (9.0 * 8.0)
        if self.vel < 5:
            driftAmount = 0

        if self.turningLeft:
            self.direction = self.direction.rotate(radiansToAngle(self.turningRate) * multiplier)

            self.driftMomentum -= driftAmount
        elif self.turningRight:
            self.direction = self.direction.rotate(-radiansToAngle(self.turningRate) * multiplier)
            self.driftMomentum += driftAmount
        self.acc = 0
        if self.accelerating:
            if self.vel < 0:
                self.acc = 3 * self.accelerationSpeed
            else:
                self.acc = self.accelerationSpeed
        elif self.reversing:
            if self.vel > 0:
                self.acc = -3 * self.accelerationSpeed
            else:
                self.acc = -1 * self.accelerationSpeed

    """
    checks every wall and if the car has hit a wall returns true    
    """

    def hitAWall(self):
        for wall in self.walls:
            if wall.hitCar(self):
                return True

        return False

    """
    returns the point of collision of a line (x1,y1,x2,y2) with the walls, 
    if multiple walls are hit it returns the closest collision point
    """

    def getCollisionPointOfClosestWall(self, x1, y1, x2, y2):
        global vec2
        minDist = 2 * displayWidth
        closestCollisionPoint = vec2(0, 0)
        for wall in self.walls:
            collisionPoint = getCollisionPoint(x1, y1, x2, y2, wall.x1, wall.y1, wall.x2, wall.y2)
            if collisionPoint is None:
                continue
            if dist(x1, y1, collisionPoint.x, collisionPoint.y) < minDist:
                minDist = dist(x1, y1, collisionPoint.x, collisionPoint.y)
                closestCollisionPoint = vec2(collisionPoint)
        return closestCollisionPoint

    """
    by creating lines in many directions from the car and getting the closest collision point of that line
    we create  "vision vectors" which will allow the car to 'see' 
    kinda like a sonar system
    """

    def getState(self):
        self.setVisionVectors()
        normalizedVisionVectors = [1 - (max(1.0, line) / self.vectorLength) for line in self.collisionLineDistances]

        normalizedForwardVelocity = max(0.0, self.vel / self.maxSpeed)
        normalizedReverseVelocity = max(0.0, self.vel / self.maxReverseSpeed)
        if self.driftMomentum > 0:
            normalizedPosDrift = self.driftMomentum / 5
            normalizedNegDrift = 0
        else:
            normalizedPosDrift = 0
            normalizedNegDrift = self.driftMomentum / -5

        normalizedAngleOfNextGate = (get_angle(self.direction) - get_angle(self.directionToRewardGate)) % 360
        if normalizedAngleOfNextGate > 180:
            normalizedAngleOfNextGate = -1 * (360 - normalizedAngleOfNextGate)

        normalizedAngleOfNextGate /= 180

        normalizedState = [*normalizedVisionVectors, normalizedForwardVelocity, normalizedReverseVelocity,
                           normalizedPosDrift, normalizedNegDrift, normalizedAngleOfNextGate]
        return np.array(normalizedState)

    def setVisionVectors(self):
        h = self.height - 4
        w = self.width
        self.collisionLineDistances = []
        self.lineCollisionPoints = []
        self.setVisionVector(w / 2, 0, 0)
        self.setVisionVector(w / 2, -h / 2, -180 / 16)
        self.setVisionVector(w / 2, -h / 2, -180 / 4)
        self.setVisionVector(w / 2, -h / 2, -4 * 180 / 8)

        self.setVisionVector(w / 2, h / 2, 180 / 16)
        self.setVisionVector(w / 2, h / 2, 180 / 4)
        self.setVisionVector(w / 2, h / 2, 4 * 180 / 8)

        self.setVisionVector(-w / 2, -h / 2, -6 * 180 / 8)
        self.setVisionVector(-w / 2, h / 2, 6 * 180 / 8)
        self.setVisionVector(-w / 2, 0, 180)

    """
    calculates and stores the distance to the nearest wall given a vector 
    """

    def setVisionVector(self, startX, startY, angle):
        collisionVectorDirection = self.direction.rotate(angle)
        collisionVectorDirection = collisionVectorDirection.normalize() * self.vectorLength
        startingPoint = self.getPositionOnCarRelativeToCenter(startX, startY)
        collisionPoint = self.getCollisionPointOfClosestWall(startingPoint.x, startingPoint.y,
                                                             startingPoint.x + collisionVectorDirection.x,
                                                             startingPoint.y + collisionVectorDirection.y)
        if collisionPoint.x == 0 and collisionPoint.y == 0:
            self.collisionLineDistances.append(self.vectorLength)
        else:
            self.collisionLineDistances.append(
                dist(startingPoint.x, startingPoint.y, collisionPoint.x, collisionPoint.y))
        self.lineCollisionPoints.append(collisionPoint)

    """
    shows dots where the collision vectors detect a wall 
    """

    def showCollisionVectors(self):
        global drawer
        for point in self.lineCollisionPoints:
            drawer.setColor([255, 0, 0])
            drawer.circle(point.x, point.y, 5)
