import random
import functions
import numpy as np
import matplotlib.pyplot as plt

def fitness(func, best_species, precision):

    if (len(best_species) > 1 and
            np.linalg.norm(np.array(best_species[-1]) - np.array(best_species[-2])) < precision):
        return False
    else:
        return True


def roulette_selection(population, func_to_optimize, population_limit):
    
    if (len(population) > population_limit):
        pop_len = len(population)
        sort = False

        while(not sort):
            sort = True
            for i in range(pop_len - 1) if pop_len < population_limit else range(population_limit - 1):
                for j in range(i + 1, pop_len):
                    if (func_to_optimize(population[i]) > func_to_optimize(population[j])):
                        tmp = population[i]
                        population[i] = population[j]
                        population[j] = tmp
                        sort = False

        
        return population[0:population_limit - 1]
    else: return population


def tournament_selection(population, func_to_optimize, population_limit):
    new_population = []

    for i in range(population_limit) if population_limit < len(population) else range(len(population)):

        first_rand = random.randint(0, len(population) - 1)
        second_rand = random.randint(0, len(population) - 1)
        if (func_to_optimize(population[first_rand]) < func_to_optimize(population[second_rand])):
            new_population.append(population[first_rand])
            population.remove(population[first_rand])
        else: 
            new_population.append(population[second_rand])
            population.remove(population[second_rand])

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


def mutate(population, ranges, mutation_probability, mutate_coef=0.01, precision = 1e-2):
    mutated_population = population[::]
    dimension = len(population[0])

    for i in range(len(mutated_population)):
        if (random.random() < mutation_probability):

            for j in range(dimension):
                dim_mutate_coef = mutate_coef / abs(ranges[j * dimension] - ranges[j * dimension - 1])
                # mutate gene in range of function
                while(True):
                    mutated_gene = population[i][j] + (ranges[j*dimension] - ranges[j*dimension - 1])*\
                                   (random.random()*dim_mutate_coef*(-1 if random.random() > 0.5 else 1))
                    if not (mutated_gene < ranges[j*dimension] or mutated_gene > ranges[j*dimension + 1]):
                        population[i][j] = mutated_gene
                        break

    return mutated_population


def find_better_species(population, func_to_optimize):

    minIdx = 0
    for i in range(len(population)):
        if (func_to_optimize(population[minIdx]) > func_to_optimize(population[i])):
            minIdx = i
    return population[minIdx]


def gen_alg(crossover_func, mutation_func, selection_func,
                func_to_optimize, dimension, function_ranges,
                crossover_probability=0.8, mutation_pobability=0.1,
                fitness_func=fitness, initial_population=100, population_limit=30, precision=1e-2):

    if not (0 <= crossover_probability <= 1 and 0 <= mutation_pobability <= 0.1 and precision > 0 and precision < 1
                and initial_population > 0 and population_limit > 0 and dimension > 0):
        raise AttributeError('Wrong probabilities')

    ## Initial population
    population = [[(function_ranges[j*dimension - 1] - function_ranges[j*dimension]) * random.random() + function_ranges[j*dimension] for j in range(dimension)]
                    for i in range(initial_population)]


    best_species = []

    i = 0
    while(fitness_func(func_to_optimize, best_species, precision)):

        population = selection_func(population, func_to_optimize, population_limit)

        population = crossover_func(population, crossover_probability)

        if (len(best_species) > 1 and best_species[-1] != best_species[-2]):
            mutate_coef = (np.linalg.norm(np.array(best_species[-1]) - np.array(best_species[-2])))
        else:
            mutate_coef = 1
        population = mutation_func(population, function_ranges, mutation_pobability, 
        precision=precision, mutate_coef=mutate_coef)

        best_species.append(find_better_species(population, func_to_optimize))
        i = i + 1

    return find_better_species(population, func_to_optimize), population, best_species, i

