import random
import functions
import numpy as np
import matplotlib.pyplot as plt

def fitness_sort(population, func):
    ## Bubble sort in order of least to greatest value of (func(population[i]))
    ## where (population[i]) is a species of population
    pop_len = len(population)
    continue_iterations = False     ## Check if there is found already
    sort = False

    while(not sort):
        sort = True
        for i in range(pop_len - 1):
            for j in range(i + 1, pop_len):
                if (func(population[i]) < func(population[j])):
                    tmp = population[i]
                    population[i] = population[j]
                    population[j] = tmp
                    sort = False
                    continue_iterations = True
    
    return continue_iterations


def roulette_selection(population, func_to_optimize, population_limit):
    ## returns slice of population list because it is already sorted in order of optimal value due to the fitness function
    if (len(population) > population_limit):
        return population[0:population_limit - 1]
    else: return population


def tournament_selection(population, func_to_optimize, population_limit):
    new_population = []
    population_len = len(population)

    for i in range(population_limit):
        first_rand = random.randint(0, population_len - 1)
        second_rand = random.randint(0, population_len - 1)
        if (func(population[first_rand]) < func(population[second_rand])):
            new_population.append(population[first_rand])
        else: new_population.append(population[second_rand])
    
    return new_population


def two_point_crossover(population, crossover_probability):
    new_population = population[::]
    start_len = len(population)
    dimension = len(population[0])

    for i in range(start_len - 1):
        for j in range(i + 1, start_len):
            firstPoint = random.randint(0, dimension)
            secondPoint = random.randint(0, dimension)
            
            if (firstPoint > secondPoint): ## if first > second then swap them
                firstPoint, secondPoint = secondPoint, firstPoint

            first_one = population[i][0:firstPoint]+population[j][firstPoint:secondPoint]+population[i][secondPoint:dimension]
            second_one = population[j][0:firstPoint]+population[i][firstPoint:secondPoint]+population[j][secondPoint:dimension]
            
            if (random.random() <= crossover_probability):
                new_population.append(first_one)

            if (random.random() <= crossover_probability and dimension > 1):
                new_population.append(second_one)

    return new_population


def uniform_crossover(population, crossover_probability):
    new_population = population[::]
    start_len = len(population)
    dimension = len(population[0])

    for i in range(start_len - 1):
        for j in range(i + 1, start_len):
            distribution = np.random.choice(a=(True, False), size=dimension)

            first_one = [population[i][k] if distribution[k] else population[j][k] for k in range(dimension)]
            second_one = [population[j][k] if distribution[k] else population[i][k] for k in range(dimension)]

            if (random.random() <= crossover_probability):
                new_population.append(first_one)
                
            if (random.random() <= crossover_probability):
                new_population.append(second_one)

    return new_population


def mutate(population, ranges, mutation_probability):
    mutated_population = population[::]
    dimension = len(population[0])

    for i in range(len(mutated_population)):
        for j in range(dimension):
            if (random.random() < mutation_probability):
                mutated_population[i][j] = (ranges[j*dimension - 1] - ranges[j*dimension]) * random.random() + ranges[j*dimension]
    
    return mutated_population


def gen_alg(crossover_func, mutation_func, selection_func,
                func_to_optimize, dimension, function_ranges,
                crossover_probability=0.5, mutation_pobability=0.1,
                fitness_func=fitness_sort, initial_population=5, population_limit=5, iteration_limit=100):

    if not (0 <= crossover_probability <= 1 and 0 <= mutation_pobability <= 0.1 and iteration_limit > 0
                and initial_population > 0 and population_limit > 0 and dimension > 0):
        raise AttributeError('Wrong probabilities')

    ## Initial population
    population = [[(ranges[j*dimension - 1] - ranges[j*dimension]) * random.random() + ranges[j*dimension] for j in range(dimension)]
                    for i in range(initial_population)]


    i = 0
    while(i < iteration_limit and fitness_func(population, func_to_optimize)):

        population = selection_func(population, func_to_optimize, population_limit)

        population = crossover_func(population, crossover_probability)

        population = mutation_func(population, ranges, mutation_pobability)
        
        print(i)
        i = i + 1

    print(i, " ", "iterations")
    minIdx = 0
    for i in range(len(population)):
        if (func(population[minIdx]) > func(population[i])):
            minIdx = i

    return population[minIdx], population

###OPTIONS      parameter name              example
##--------------------------------------------------------------------------------
## FUNCTION     (function_to_optimize):     functions.first(), functions.third(), 
#                                           functions.fifth(), functions.eighth(), functions.twelfth()
##
## CROSSOVER    (crossover_func):           two_point_crossover, uniform_crossover
##
## SELECTION    (selection_func):           roulette_selection, tournament_selection
##
## MUTATION     (mutation_func):            mutate
##
## MUTATION_P   (mutation_probability):     0 <= x <= 0.1
##
## CROSSOVER_P  (crossover_probability):    0 <= x <= 1
##--------------------------------------------------------------------------------
################# CHANGE HERE
FUNCTION = functions.fifth()
CROSSOVER = uniform_crossover
SELECTION = tournament_selection
MUTATION_PROBABILITY = 0.1
CROSSOVER_PROBABILITY = 0.5
INITIAL_POPULATION = 30
POPULATION_LIMIT = 10
ITERATION_LIMIT = 100
#################################################
func, dimension, ranges, print_plot = FUNCTION
pop, population = gen_alg(crossover_func=CROSSOVER, mutation_func=mutate, selection_func=SELECTION,
                            func_to_optimize=func, dimension=dimension, function_ranges=ranges,
                            crossover_probability=CROSSOVER_PROBABILITY, mutation_pobability=MUTATION_PROBABILITY,
                            fitness_func=fitness_sort, initial_population=INITIAL_POPULATION, 
                            population_limit=POPULATION_LIMIT, iteration_limit=ITERATION_LIMIT)


### SHOWING RESULTS
print(pop)
print(func(pop))

print_plot()
if (func != functions.__first):
    for p in population:
        plt.plot(p[0], p[1], 'wo')
    plt.plot(pop[0], pop[1], 'ro')
else:
    for p in population:
        plt.plot(p[0], func(p), 'go')
    plt.plot(pop[0], func(pop), 'ro')

plt.show()
