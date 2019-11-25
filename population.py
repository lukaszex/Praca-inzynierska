from city_fitness import *
import random
import operator
import numpy as np
import pandas as pd
import collections
from matplotlib import pyplot as plt

mutations1 = 0
mutations2 = 0
mutations3 = 0

class Population:
    def __init__ (self, populationID, routesList, populationSize):
        self.populationID = populationID
        self.population = routesList
        self.populationSize = populationSize
        self.bestValue = 0
        self.stdDev = 0
        self.routeLengths = []
        self.secondValue = 0
        self.thirdValue = 0
        self.averageValue = 0
        self.medianValue = 0
        self.worstValue = 0

    def orderRoutes (self):
        fitnessRoutes = {}
        for i in range(len(self.population)):
            fitnessRoutes[self.population[i]] = self.population[i].calculateFitness()
        self.population = sorted(fitnessRoutes.items(), key = operator.itemgetter(1), reverse = True)

    def calculateLengths (self):
        for i in range(len(self.population)):
            self.routeLengths.append(self.population[i][0].routeLength)
        self.routeLengths.sort()
        self.bestValue = self.routeLengths[0]
        self.secondValue = self.routeLengths[1]
        self.thirdValue = self.routeLengths[2]
        self.averageValue = np.average(self.routeLengths)
        self.stdDev = np.std(self.routeLengths)
        self.medianValue = np.median(self.routeLengths)
        self.worstValue = self.routeLengths[self.populationSize - 1]


def createRoute (cities):
    route = random.sample(cities, len(cities))
    return Route(route)

def createInitialPopulation (populationSize, cities):
    routesList = []
    for i in range(0, populationSize):
        routesList.append(createRoute(cities))
    return Population(1, routesList, populationSize)

def selection (population, eliteNumber):
    selectedRoutes = []
    df = pd.DataFrame(np.array(population.population), columns = ['Route', 'Fitness'])
    #population.stdDev = df.loc[:, 'Route'].std()
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()
    for i in range(0, eliteNumber):
        selectedRoutes.append(population.population[i][0])
    i = 0
    while i < len(population.population) - eliteNumber:
        rand = 100 * random.random()
        j = 0
        while df.iat[j, 3] < rand:
            j += 1
        selectedRoutes.append(population.population[j][0])
        i += 1
    return selectedRoutes

def crossover (parent1, parent2):
    child = []
    childL = []
    childP = []
    gene1 = int(random.random() * len(parent1))
    gene2 = int(random.random() * len(parent1))
    startGene = min(gene1, gene2)
    endGene = max(gene1, gene2)
    for i in range(startGene, endGene):
        childL.append(parent1[i])
    childP = [item for item in parent2 if item not in childL]
    child = childL + childP
    return Route(child)

def breed (selectedRoutes, eliteNumber):
    children = []
    for i in range(0, eliteNumber):
        children.append(selectedRoutes[i])
    for i in range(eliteNumber, len(selectedRoutes)):
        children.append(crossover(selectedRoutes[i - eliteNumber].route, selectedRoutes[len(selectedRoutes) - i - eliteNumber - 1].route))
    return children

def mutate1 (route, mutationProbability):
    global mutations1
    for i in range(len(route)):
        if random.random() < mutationProbability:
            mutations1 += 1
            j = int(random.random() * len(route))
            city1 = route[i]
            city2 = route[j]
            route[i] = city2
            route[j] = city1
    return Route(route)

def mutate2 (route, mutationProbability):
    global mutations2
    for i in range(len(route)):
        if random.random() < mutationProbability:
            mutations2 += 1
            j = int(random.random() * len(route))
            k = int(random.random() * len(route))
            city1 = route[i]
            city2 = route[j]
            city3 = route[k]
            route[i] = city3
            route[j] = city1
            route[k] = city2
        return Route(route)

def mutate3 (route, mutationProbability):
    global mutations3
    for i in range(len(route)):
        if random.random() < mutationProbability:
            mutations3 += 1
            city = route[i]
            del(route[i])
            j = int(random.random() * len(route))
            route.insert(j, city)
    return Route(route)

def mutatePopulation (children, eliteNumber, mutationProbability):
    mutatedChildren = []
    numbers = [1, 2, 3]
    for i in range(0, eliteNumber):
        mutatedChildren.append(children[i])
    for i in range(eliteNumber, len(children)):
        nr = random.choice(numbers)
        if nr == 1:
            mutatedChildren.append(mutate1(children[i].route, mutationProbability))
        elif nr == 2:
            mutatedChildren.append(mutate2(children[i].route, mutationProbability))
        elif nr == 3:
            mutatedChildren.append(mutate3(children[i].route, mutationProbability))
    return mutatedChildren

def nextGeneration (population, eliteNumber, mutationProbability):
    #population.orderRoutes()
    selectedRoutes = selection(population, eliteNumber)
    children = breed(selectedRoutes, eliteNumber)
    mutatedChildren = mutatePopulation(children, eliteNumber, mutationProbability)
    newPopulation = Population(population.populationID + 1, mutatedChildren, population.populationSize)
    newPopulation.orderRoutes()
    return newPopulation

def runAlgorithm (cities, populationSize, eliteNumber, mutationProbability, generations):
    bestValues = []
    stdDevs = []
    averages = []
    medians = []
    secondValues = []
    thirdValues = []
    worstValues = []
    population = createInitialPopulation(populationSize, cities)
    population.orderRoutes()
    for i in range(0, generations):
        population.calculateLengths()
        print("ID: " + str(population.populationID) + " Distance: " + str(population.bestValue))
        bestValues.append(population.bestValue)
        stdDevs.append(population.stdDev)
        averages.append(population.averageValue)
        medians.append(population.medianValue)
        secondValues.append(population.secondValue)
        thirdValues.append(population.thirdValue)
        worstValues.append(population.worstValue)
        population = nextGeneration(population, eliteNumber, mutationProbability)
    #population.orderRoutes()
    print("ID: " + str(population.populationID) + "Final distance: " + str(population.population[0][0].routeLength) + " Route:" + str(population.population[0][0].route) + "\nmutation1: " + str(mutations1) + "\nmutation2: " + str(mutations2) + "\nmutation3: " + str(mutations3))
    plt.figure(1)
    plt.plot(bestValues)
    plt.xlabel('Pokolenie')
    plt.ylabel('Wartość funkcji celu najlepszego osobnika')
    plt.grid()
    plt.show()
    plt.figure(2)
    plt.plot(averages, color = 'skyblue')
    plt.plot(medians, color = 'red')
    plt.xlabel('Pokolenie')
    plt.ylabel('Średnia i mediana wartości funkcji celu')
    plt.grid()
    plt.show()
    plt.figure(3)
    plt.plot(stdDevs)
    plt.xlabel('Pokolenie')
    plt.ylabel('Odchylenie standardowe wartości funkcji celu')
    plt.grid()
    plt.show()
    plt.figure(4)
    plt.plot(bestValues, color = 'skyblue')
    plt.plot(secondValues, color = 'green')
    plt.plot(thirdValues, color = 'purple')
    plt.plot(worstValues, color = 'red')
    plt.xlabel('Pokolenie')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    cities = readData("test1")
    #for i in range(1, 25):
    #    cities.append(City(i, int(100 * random.random()), int(100 * random.random())))
    runAlgorithm(cities, 100, 10, 0.02, 2000)

