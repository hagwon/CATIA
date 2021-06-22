import math

class inversKinematics():

    # default
    # link1 = 64
    # link2 = 44
    # link3 = 210

    def __init__(self, link1=64, link2=44, link3=210):
        self.link1 = link1
        self.link2 = link2
        self.link3 = link3
        self.last_step = [0, 0, 0]

        self.a = 0
        self.b = 0
        self.c = 0
        self.d = 0
        self.pi = 0
        self.cos_a = 0
        self.cos_b = 0
        self.theta = [0, 0, 0]
        self.angle = [0, 0, 0]
    
    def inputDimension(self, index, dimension):
        if (index == 0):
            self.link1 = dimension
        elif (index == 1):
            self.link2 = dimension
        elif (index == 2):
            self.link3 = dimension
    
    def invers(self, x, y, z):

        self.c = math.pow(x, 2) + math.pow(y, 2)
        self.d = math.sqrt(self.c + math.pow(z - self.link1, 2))
        self.cos_a = (math.pow(self.link2, 2) + math.pow(self.d, 2) - math.pow(self.link3, 2)) / (2 * self.link2 * self.d)
        self.cos_b = (math.pow(self.link2, 2) + math.pow(self.link3, 2) - math.pow(self.d, 2)) / (2 * self.link2 * self.link3)

        if (-1 <= self.cos_a and self.cos_a <= 1) or (-1 <= self.cos_b and self.cos_b <= 1):
            self.pi = math.atan2(z - self.link1, math.sqrt(self.c))
            self.a = math.atan2(math.sqrt(1 - math.pow(self.cos_a, 2)), self.cos_a)
            self.b = math.atan2(math.sqrt(1 - math.pow(self.cos_b, 2)), self.cos_b)

            self.theta[0] = math.atan2(y, x)
            self.theta[1] = math.pi / 2 - self.pi - self.a
            self.theta[2] = math.pi - self.b

            self.angle[0] = math.degrees(self.theta[0])
            self.angle[1] = math.degrees(self.theta[1])
            self.angle[2] = math.degrees(self.theta[2])