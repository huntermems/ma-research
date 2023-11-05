import numpy as np
from itertools import chain
import random
from config import get_order_type, total_t
from collections import defaultdict 

tree = lambda: defaultdict(tree)

class AntColonyOptimization:

    # """ numAnts: Integer
    #     numInterations: Integer
    #     pheromoneWeight: Float
    #     heuristicWeight: Float
    #     evaporationRate: Float
    #     order: Array char[] 
    #     itemMapping: dict
    # """


    def __init__(self, numAnts=100, numIterations=100, pheromoneWeight=1.0, heuristicWeight=2.0, evaporationRate=1.0, order=[], itemMapping={}, randonInstance=random.SystemRandom()):
        self.numAnts = numAnts
        self.numIterations = numIterations
        self.evaporationRate = evaporationRate
        self.pheromoneWeight = pheromoneWeight
        self.heuristicWeight = heuristicWeight
        self.order = order
        self.orderLength = len(order)
        self.itemMapping = itemMapping
        self.randomInstance = randonInstance
        self.distanceMapping = tree()
        self.pheromoneMapping = tree()
    
    def computeDistance(self, city1, city2):
        distance = total_t([city1, city2], False)
        return distance

    def run(self):
        bestPath = None
        bestPathLength = np.inf
        for _ in range(self.numIterations):
            paths = []
            pathLengths = []
            for ant in range(self.numAnts):
               path = self.constructPath()
               pathLength = self.getPathLength(path)
               paths.append(path)
               pathLengths.append(pathLength)
               if pathLength < bestPathLength:
                bestPath = path
                bestPathLength = pathLength
            
            self.updatePheromone(paths, pathLengths)
        
        return bestPath, bestPathLength
    
    def getPathLength(self, path):
        return total_t(path)
    
    def constructPath(self):
        visitedCities = []
        visitedCityTypes = []
        # An ant chooses a random cities
        cityPool = self.getCityPool(self.order)
        firstCity = self.randomInstance.choice(cityPool)
        firstCityType = get_order_type(firstCity)
        # Add the city to the visted
        visitedCities.append(firstCity)
        visitedCityTypes.append(firstCityType)
        
        while len(visitedCityTypes) < self.orderLength:
            nextCity = self.getNextCity(firstCity, visitedCityTypes)
            nextCityType = get_order_type(nextCity)
            visitedCities.append(nextCity)
            visitedCityTypes.append(nextCityType)

        return visitedCities

    def getCityPool(self, itemTypes):
        cityPool = list(chain.from_iterable([value for key, value in self.itemMapping.items() if key in itemTypes]))
        return cityPool

    def getNextCity(self, currentCity, visitedCityTypes):
        citiesNeedToBeVisited = self.order.copy()
        for city in visitedCityTypes:
            citiesNeedToBeVisited.remove(city)
        cityPool = self.getCityPool(citiesNeedToBeVisited)
        probabilities = self.calculateProbabilities(currentCity, cityPool)
        if all(c == 0 for c in probabilities):
            return self.randomInstance.choices(cityPool)[0]
        return self.randomInstance.choices(cityPool, probabilities)[0]
    
    def calculateProbabilities(self, currentCity, cityPool):
        pheromoneSum = 0.0
        pheromoneCopy = self.pheromoneMapping.copy()
        for city in cityPool:
            pheromone = 0
            if pheromoneCopy[currentCity][city] and (isinstance(pheromoneCopy[currentCity][city], float) or isinstance(pheromoneCopy[currentCity][city], int)):
                pheromone = pheromoneCopy[currentCity][city] ** self.pheromoneWeight
            
            heuristic = (1.0 / self.computeDistance(currentCity, city)) ** self.heuristicWeight
            pheromoneSum += pheromone * heuristic
            

        probabilities = []

        for city in cityPool:
            pheromone = 0
            if pheromoneCopy[currentCity][city] and (isinstance(pheromoneCopy[currentCity][city], float) or isinstance(pheromoneCopy[currentCity][city], int)):
                pheromone = pheromoneCopy[currentCity][city] ** self.pheromoneWeight

            heuristic = (1.0 / self.computeDistance(currentCity, city)) ** self.heuristicWeight
            
            probability = (pheromone * heuristic) / pheromoneSum if pheromoneSum else 0
            probabilities.append(probability)


        return probabilities
    
    def updatePheromone(self, paths, pathLengths):
        
        for i, path in enumerate(paths):
            for j in range(len(path) - 1):
                city1 = path[j]
                city2 = path[j + 1]
                self.pheromoneMapping[city1][city2] = 0
                self.pheromoneMapping[city1][city2] += (1.0 - self.evaporationRate) * 1.0 / pathLengths[i]

    def printPool(self):
        cityPool = self.getCityPool(self.order)
        print(cityPool)
        print(len(cityPool))