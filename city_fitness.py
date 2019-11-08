import numpy as np

class City:
    def __init__ (self, cityID, x, y):
        self.cityID = cityID
        self.x = x
        self.y = y

    def calculateDistance (self, city):
        return np.sqrt ((abs(self.x - city.x) ** 2) + (abs(self.y - city.y) ** 2))

    def __repr__ (self):
        return "ID: " + str(self.cityID) + " (" + str(self.x) + ", " + str(self.y) + ")"

class Route:
    def __init__ (self, route):
        self.route = route
        self.routeLength = 0
        self.fitness = 0
        self.timeAlive = 0

    def calculateRouteLength (self):
        if self.routeLength == 0:
            for i in range(len(self.route)):
                startCity = self.route[i]
                if i + 1 < len(self.route):
                    destCity = self.route[i + 1]
                else:
                    destCity = self.route[0]
                self.routeLength += startCity.calculateDistance(destCity)
        return self.routeLength

    def calculateFitness (self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.calculateRouteLength())
        return self.fitness

    def __repr__ (self):
        return str(self.route) + str(self.routeLength)
