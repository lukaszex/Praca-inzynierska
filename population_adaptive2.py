from city_fitness import *
import random
import operator
import numpy as np
import pandas as pd
import collections
from matplotlib import pyplot as plt

mutations1 = [0] * 1001
mutations2 = [0] * 1001
mutations3 = [0] * 1001
mutations = [0] * 1001

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
        self.mut1probs = []
        self.mut2probs = []
        self.mut3probs = []
        self.mut1avg = 0
        self.mut2avg = 0
        self.mut3avg = 0

    def orderRoutes (self):
        #fitnessRoutes = {}
        fitnesses = []
        for i in range(len(self.population)):
            fitnesses.append(self.population.iat[i, 0].calculateFitness())
        self.population['Fitness'] = fitnesses
        self.population = self.population.sort_values(by = ['Fitness'], ascending = False)

    def calculateLengths (self):
        for i in range(len(self.population)):
            self.routeLengths.append(self.population.iat[i, 0].routeLength)
        self.routeLengths.sort()
        self.bestValue = self.routeLengths[0]
        self.secondValue = self.routeLengths[1]
        self.thirdValue = self.routeLengths[2]
        self.averageValue = np.average(self.routeLengths)
        self.stdDev = np.std(self.routeLengths)
        self.medianValue = np.median(self.routeLengths)
        self.worstValue = self.routeLengths[self.populationSize - 1]
        for i in range(len(self.population)):
            self.mut1probs.append(self.population.iat[i, 1][0])
            self.mut2probs.append(self.population.iat[i, 1][1])
            self.mut3probs.append(self.population.iat[i, 1][2])
        self.mut1avg = np.average(self.mut1probs)
        self.mut2avg = np.average(self.mut2probs)
        self.mut3avg = np.average(self.mut3probs)


def createRoute (cities):
    route = random.sample(cities, len(cities))
    return Route(route)

def createInitialPopulation (populationSize, cities):
    routesList = []
    parameters = []
    for i in range(0, populationSize):
        routesList.append(createRoute(cities))
    for i in range(0, populationSize):
        parameters_tmp = []
        for j in range(0, 3):
            parameters_tmp.append(random.random()/50)
        parameters.append(parameters_tmp)
    d = {'route': routesList, 'params' : parameters}
    df = pd.DataFrame(d)
    return Population(1, df, populationSize)

def selection (population, eliteNumber):
    selectedRoutes = {}
    df = population.population
    #population.stdDev = df.loc[:, 'Route'].std()
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()
    for i in range(0, eliteNumber):
        selectedRoutes[population.population.iat[i, 0]] = population.population.iat[i, 1]
    i = 0
    while len(selectedRoutes) < population.populationSize:
        rand = 100 * random.random()
        j = 0
        while df.iat[j, 4] < rand:
            j += 1
        selectedRoutes[population.population.iat[j, 0]] = population.population.iat[j, 1]
        i += 1
    return selectedRoutes

def crossover (parent1, parent2):
    parent1route = parent1[0].route
    parent2route = parent2[0].route
    parent1params = parent1[1]
    parent2params = parent2[1]
    childroute = []
    childL = []
    childP = []
    childparams = []
    gene1 = int(random.random() * len(parent1route))
    gene2 = int(random.random() * len(parent1route))
    startGene = min(gene1, gene2)
    endGene = max(gene1, gene2)
    for i in range(startGene, endGene):
        childL.append(parent1route[i])
    childP = [item for item in parent2route if item not in childL]
    childroute = childL + childP
    childroute = Route(childroute)
    breakGene = random.choice([0, 1])
    childparams = parent1params[:breakGene] + parent2params[breakGene:]
    return [childroute, childparams]


def breed (selectedRoutes, eliteNumber):
    children = []
    #childrenTmp = []
    selectedRoutes = list(selectedRoutes.items())
    for i in range(0, eliteNumber):
        #children[selectedRoutes[i][0]] = selectedRoutes[i][1]
        children.append(selectedRoutes[i])
    for i in range(eliteNumber, len(selectedRoutes)):
        children.append(crossover(selectedRoutes[i - eliteNumber], selectedRoutes[len(selectedRoutes) - i - eliteNumber - 1]))
    return children

def mutateParams (route):
    params = route[1]
    for i in params:
        if random.random() < i:
            rand = random.choice([0, 1])
            if rand == 0:
                i += random.random()/500
            else:
                i -= random.random()/500
    route[1] = params
    return route

def mutate1 (route, population):
    global mutations1
    global mutations
    routeTmp = route[0]
    routeParams = route[1]
    for i in range(len(routeTmp.route)):
        if random.random() < routeParams[0]:
            mutations1[population.populationID] += 1
            mutations[population.populationID] += 1
            j = int(random.random() * len(routeTmp.route))
            city1 = routeTmp.route[i]
            city2 = routeTmp.route[j]
            routeTmp.route[i] = city2
            routeTmp.route[j] = city1
    route[0] = routeTmp
    return route

def mutate2 (route, population):
    global mutations2
    global mutations
    routeTmp = route[0]
    routeParams = route[1]
    for i in range(len(routeTmp.route)):
        if random.random() < routeParams[1]:
            mutations2[population.populationID] += 1
            mutations[population.populationID] += 1
            j = int(random.random() * len(routeTmp.route))
            k = int(random.random() * len(routeTmp.route))
            city1 = routeTmp.route[i]
            city2 = routeTmp.route[j]
            city3 = routeTmp.route[k]
            routeTmp.route[i] = city3
            routeTmp.route[j] = city1
            routeTmp.route[k] = city2
    route[0] = routeTmp
    return route

def mutate3 (route, population):
    global mutations3
    global mutations
    routeTmp = route[0]
    routeParams = route[1]
    for i in range(len(routeTmp.route)):
        if random.random() < routeParams[2]:
            mutations3[population.populationID] += 1
            mutations[population.populationID] += 1
            city = routeTmp.route[i]
            del(routeTmp.route[i])
            j = int(random.random() * len(routeTmp.route))
            routeTmp.route.insert(j, city)
    route[0] = routeTmp
    return route

def mutatePopulation (children, eliteNumber, population):
    mutatedChildren = []
    numbers = [1, 2, 3]
    for i in range(0, eliteNumber):
        mutatedChildren.append(children[i])
    for i in range(eliteNumber, len(children)):
        mutateParams(children[i])
    for i in range(eliteNumber, len(children)):
        #nr = children[i][1].index(max(children[i][1])) + 1
        nr = random.choices(numbers, weights = [children[i][1][0], children[i][1][1], children[i][1][2]])
        if nr[0] == 1:
            mutatedChildren.append(mutate1(children[i], population))
        elif nr[0] == 2:
            mutatedChildren.append(mutate2(children[i], population))
        elif nr[0] == 3:
            mutatedChildren.append(mutate3(children[i], population))
    df = pd.DataFrame(mutatedChildren, columns = ['route', 'params'])
    return df

def nextGeneration (population, eliteNumber, mutationProbability):
    #population.orderRoutes()
    selectedRoutes = selection(population, eliteNumber)
    children = breed(selectedRoutes, eliteNumber)
    mutatedChildren = mutatePopulation(children, eliteNumber, population)
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
    mut1avgs = []
    mut2avgs = []
    mut3avgs = []
    global mutations
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
        mut1avgs.append(population.mut1avg)
        mut2avgs.append(population.mut2avg)
        mut3avgs.append(population.mut3avg)
        population = nextGeneration(population, eliteNumber, mutationProbability)
    #population.orderRoutes()
    print("ID: " + str(population.populationID) + "Final distance: " + str(population.bestValue) + " Route:" + str(population.population.iat[0, 0].route) + "\nmutation1: " + str(mut1avgs[999]) + "\nmutation2: " + str(mut2avgs[999]) + "\nmutation3: " + str(mut3avgs[999]) +
          "\nmutation1best: " + str(population.population.iat[0, 1][0]) + "\nmutation2best: " + str(population.population.iat[0, 1][1]) + "\nmutation3best: " + str(population.population.iat[0,1][2]))
    devsSeries = pd.Series(stdDevs)
    rolling_mean = devsSeries.rolling(window=50).mean()
    plt.figure(3)
    plt.plot(stdDevs, label='Odchylenie standardowe wartości funkcji celu')
    plt.plot(rolling_mean, color='black', label='Średnia krocząca')
    plt.xlabel('Pokolenie')
    plt.ylabel('Odchylenie standardowe wartości funkcji celu')
    plt.legend()
    plt.grid()
    plt.show()
    plt.figure(4)
    plt.plot(bestValues, color='skyblue', label='Najlepsza wartość funkcji celu')
    plt.plot(averages, color='green', label='Średnia wartość funkcji celu')
    # plt.plot(thirdValues, color = 'purple')
    plt.plot(worstValues, color='red', label='Najgorsza wartość funkcji celu')
    plt.xlabel('Pokolenie')
    plt.ylabel('Średnia, najlepsza i najgorsza wartość funkcji celu')
    plt.legend()
    plt.grid()
    plt.show()
    plt.figure(5)
    plt.plot(mutations1, color = 'green', label = 'Liczba mutacji (i)')
    plt.plot(mutations2, color = 'purple', label = 'Liczba mutacji (ii)')
    plt.plot(mutations3, color = 'red', label = 'Liczba mutacji (iii)')
    plt.xlabel('Pokolenie')
    plt.ylabel('Liczba mutacji')
    plt.legend()
    plt.grid()
    plt.show()
    plt.figure(6)
    plt.plot(mut1avgs, color = 'green', label = 'Średnie prawdopodobieństwo mutacji (i)')
    plt.plot(mut2avgs, color = 'purple', label = 'Średnie prawdopodobieństwo mutacji (ii)')
    plt.plot(mut3avgs, color = 'red', label = 'Średnie prawdopodobieństwo mutacji (iii)')
    plt.xlabel('Pokolenie')
    plt.ylabel('Średnie prawdopodobieństwo mutacji')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    cities = readData("berlin52")
    #for i in range(1, 25):
    #    cities.append(City(i, int(100 * random.random()), int(100 * random.random())))
    runAlgorithm(cities, 100, 20, 0.02, 1000)

