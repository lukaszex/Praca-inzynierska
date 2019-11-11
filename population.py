from city_fitness import City, Route
import random
import operator
import numpy as np
import pandas as pd
import collections

class Population:
    def __init__ (self, populationID, routesList, populationSize):
        self.populationID = populationID
        self.population = routesList
        self.populationSize = populationSize

    def orderRoutes (self):
        fitnessRoutes = {}
        for i in range(len(self.population)):
            fitnessRoutes[self.population[i]] = self.population[i].calculateFitness()
        self.population = sorted(fitnessRoutes.items(), key = operator.itemgetter(1), reverse = True)


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
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()
    for i in range(0, eliteNumber):
        selectedRoutes.append(population.population[i][0])
    i = 0
    while i <= len(population.population) - eliteNumber:
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
        children.append(crossover(selectedRoutes[i].route, selectedRoutes[len(selectedRoutes) - i - 1].route))
    return children

def mutate (route, mutationProbability):
    for i in range(len(route)):
        if random.random() < mutationProbability:
            j = int(random.random() * len(route))
            city1 = route[i]
            city2 = route[j]
            route[i] = city2
            route[j] = city1
    return Route(route)

def mutatePopulation (children, mutationProbability):
    mutatedChildren = []
    for i in children:
        mutatedChildren.append(mutate(i.route, mutationProbability))
    return mutatedChildren

def nextGeneration (population, eliteNumber, mutationProbability):
    #population.orderRoutes()
    selectedRoutes = selection(population, eliteNumber)
    children = breed(selectedRoutes, eliteNumber)
    mutatedChildren = mutatePopulation(children, mutationProbability)
    newPopulation = Population(population.populationID + 1, mutatedChildren, population.populationSize)
    newPopulation.orderRoutes()
    return newPopulation

def runAlgorithm (cities, populationSize, eliteNumber, mutationProbability, generations):
    population = createInitialPopulation(populationSize, cities)
    population.orderRoutes()
    for i in range(0, generations):
        population = nextGeneration(population, eliteNumber, mutationProbability)
        print("Distance: " + str(population.population[0][0].routeLength))
    population.orderRoutes()
    print("Final distance: " + str(1 / population.population[0].calculateRouteLength()))

if __name__ == "__main__":
    #city1 = City (1, 0, 0)
    #city2 = City (2, 1, 1)
    #city3 = City (3, 2, 3)
    #city4 = City (4, 5, 8)
    #city5 = City (5, 2, 9)
    #city6 = City (6, 9, 1)
    #city7 = City (7, 11, 4)
    #cities = [city1, city2, city3, city4, city5, city6, city7]
    cities = []
    for i in range(1, 50):
        cities.append(City(i, int(100 * random.random()), int(100 * random.random())))
    runAlgorithm(cities, 50, 2, 0.01, 100)
    #pop = createInitialPopulation(20, cities)
    #pop.orderRoutes()
    #sel = selection (pop)
    #children = breed (sel, 2)
    #print ([item for item, count in collections.Counter(sel).items() if count > 1])
