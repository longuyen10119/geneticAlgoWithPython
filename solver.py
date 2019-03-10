import math
import sys
import time
# import matplotlib.pylot as plt
import random
from numpy.random import seed
from numpy.random import shuffle
from numpy.random import randint
from numpy.random import rand
from copy import deepcopy

cities = []
distanceMap = []
numPop = 300
population = []
order = []
bestEver = []
bestDist = 0
fitness = []
normalizedFitness = []
topPerformers = []
numTopPerformers = 10
# READ IN A FILE, PARSE IT THEN GET ALL THE CITY TO CITIES


def readFile(filename):
    infile = open(filename, 'r')
    problemName = infile.readline().strip().split(':')[1]
    print("Name = " + problemName)
    comment = infile.readline().strip().split(':')[1]
    type = infile.readline().strip().split(':')[1]
    dimension = infile.readline().strip().split(':')[1]
    edgeWeight = infile.readline().strip().split(':')[1]
    header = infile.readline()

    # Read city list
    # run for loop to get city, lat,  long from the file

    for i in range(0, int(dimension)):
        city, lat, long = infile.readline().strip().split()
        cities.append([int(city), float(lat), float(long)])
        order.append(int(city))

    infile.close()


def greedySearch():
    # Feed time and start with a random city
    currentCities = order[:]
    cityTraveled = []
    random.seed(time.time())
    r = randint(len(cities) - 1)

    # append the first city random r
    # then remove it
    cityTraveled.append(currentCities[r])
    currentCities.remove(currentCities[r])
    while (len(currentCities) != 0):
        shortestDist = math.inf
        shortestC = 0
        for c in currentCities:
            dist = lookupDistanceMap(c, cityTraveled[-1])
            if (dist < shortestDist):
                shortestDist = dist
                shortestC = c

        # append the next shortest route to cityTravel
        # then remove it from current
        cityTraveled.append(shortestC)
        currentCities.remove(shortestC)

    return cityTraveled

# function to calculate Euclidian distance between 2 cities


def calEuc(city1, city2):
    eucD = math.sqrt(pow((city1[1] - city2[1]), 2) +
                     pow((city1[2] - city2[2]), 2))
    return eucD


def lookupDistanceMap(cityA, cityB):
    bigger = cityA if cityA > cityB else cityB
    smaller = cityB if cityA > cityB else cityA
    for j in range(len(distanceMap)):
        if (distanceMap[j][0] == smaller and distanceMap[j][1] == bigger):
            return distanceMap[j][2]


def calcDistance(route):
    numCities = len(route)
    dist = 0
    for i in range(numCities-1):
        # i , i+1
        bigger = route[i] if route[i] > route[i+1] else route[i+1]
        smaller = route[i+1] if route[i] > route[i+1] else route[i]

        # loop thru distance map to find distance
        for j in range(len(distanceMap)):
            if (distanceMap[j][0] == smaller and distanceMap[j][1] == bigger):
                dist += distanceMap[j][2]
    return dist


def calTotalDistance(route):
    totalDistance = 0
    for i in range(1, len(route)):
        totalDistance += calEuc(route[i-1], route[i])
    totalDistance += calEuc(route[0], route[len(route)-1])
    return totalDistance


def mappingDistanceFunction():
    numCities = len(cities)
    for i in range(numCities-1):
        for j in range(i+1, numCities):
            dist = calEuc(cities[i], cities[j])
            distanceMap.append([cities[i][0], cities[j][0], dist])


def crossover(orderA, orderB):
    seed(int(math.floor(random.uniform(1, 1000))))
    start = randint(len(orderA)-1)
    end = randint(start+1, len(orderA))

    while (end-start < math.floor(len(orderA)*0.3)):
        start = randint(len(orderA)-1)
        end = randint(start+1, len(orderA))

    newOrder = orderA[start:end]
    leftOver = list(filter(lambda x: x not in newOrder, orderB))
    newOrder = newOrder + leftOver
    return newOrder


def calcFitness():
    recordDist = math.inf
    global bestEver
    global bestDist

    fitness.clear()
    for p in population:
        d = calcDistance(p)
        f = 1/(math.pow(d, 8)+1)
        fitness.append(f)
        if(d < recordDist):
            recordDist = d
            bestEver = deepcopy(p)
            bestDist = d


def normalizeFitness():
    sum = 0
    global normalizedFitness
    global topPerformers
    global numTopPerformers
    global fitness
    normalizedFitness.clear()
    topPerformers.clear()
    for f in fitness:
        sum += f
    for f in fitness:
        normalizedFitness.append(f/sum)
    topPerformers.clear()
    newSortedFitness = normalizedFitness[:]
    newSortedFitness.sort()
    for i in range(0, numTopPerformers):
        indexInOld = normalizedFitness.index(newSortedFitness[i])
        topPerformers.append(population[indexInOld])


def nextGeneration():
    newPop = []
    global population
    newPop.append(bestEver)
    newPop.append(bestEver)

    # for i in range(numTopPerformers):
    #     newPop.append(topPerformers[i])
    for i in range(len(population)-2):
        seed(int(math.floor(random.uniform(1, 1000))))
        indexA = randint(numTopPerformers)
        # seed(int(math.floor(random.uniform(1, 1000))))
        # indexB = randint(numTopPerformers)
        tempA = topPerformers[indexA]
        # tempB = topPerformers[indexB]
        # tempA = pickOne(population, normalizedFitness)
        tempB = pickOne(population, normalizedFitness)
        temp = crossover(tempA, tempB)
        mutate(temp, 0.01)
        newPop.append(temp)

    population.clear()
    population = deepcopy(newPop)


def mutate(order, mutationRate):
    seed(int(math.floor(random.uniform(1, 1000))))
    for i in range(len(order)):
        r = rand()
        if (r < mutationRate):
            indexA = randint(len(order))
            seed(int(math.floor(random.uniform(1, 10000))))
            indexB = randint(len(order))
            swap(order, indexA, indexB)


def swap(a, i, j):
    temp = a[i]
    a[i] = a[j]
    a[j] = temp


def pickOne(list, prob):
    index = 0
    seed(int(math.floor(random.uniform(1, 1000))))
    r = rand()
    while (r > 0):
        r = r - prob[index]
        index += 1

    index -= 1
    return list[index][:]


def main():
    global topPerformers
    readFile('berlin52.tsp')
    mappingDistanceFunction()
    newOrder = greedySearch()
    print(calcDistance(newOrder))

    # Create a Population
    # seed 1
    seed(int(math.floor(random.uniform(1, 1000))))

    for i in range(numPop):
        tempOrder = deepcopy(newOrder)
        population.append(tempOrder)
    # for i in range(numPop/2):
    #     shuffle(newOrder)
    #     tempOrder = deepcopy(newOrder)
    #     population.append(tempOrder)

    generations = 1000
    for i in range(generations):

        calcFitness()
        normalizeFitness()
        print('Generation ', i+1, bestDist, calcDistance(topPerformers[0]), calcDistance(
            topPerformers[1]), calcDistance(topPerformers[2]))
        nextGeneration()


main()
