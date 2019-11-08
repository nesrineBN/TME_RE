from deap import *

import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import random
from scipy.optimize import minimize

from plot import *

import array
import random

import numpy

from deap import algorithms
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools

IND_SIZE = 1
MIN_VALUE = -30
MAX_VALUE = 30
MIN_STRATEGY = 0.5
MAX_STRATEGY = 3


# La fonction doit être minimisée, le poids est donc de -1
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", array.array, typecode="d", fitness=creator.FitnessMin, strategy=None)
creator.create("Strategy", array.array, typecode="d")

def wrapper(x):
    value = benchmarks.schaffer_mo(x)
    return value

# Génération d'un individu avec une distribution uniforme dans les bornes indiquées
def generateES(icls, scls, size, imin, imax, smin, smax):
    ind = icls(random.uniform(imin, imax) for _ in range(size))
    ind.strategy = scls(random.uniform(smin, smax) for _ in range(size))
    return ind

# Fonction utilisée pour mettre une borne inférieure à la mutation
def checkStrategy(minstrategy):
    def decorator(func):
        def wrappper(*args, **kargs):
            children = func(*args, **kargs)
            for child in children:
                for i, s in enumerate(child.strategy):
                    if s < minstrategy:
                        child.strategy[i] = minstrategy
            return children
        return wrappper
    return decorator

## A completer: creator et toolbox de DEAP ##
toolbox = base.Toolbox()
toolbox.register("individual", generateES, creator.Individual, creator.Strategy,
                 IND_SIZE, MIN_VALUE, MAX_VALUE, MIN_STRATEGY, MAX_STRATEGY)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxESBlend, alpha=0.1)
toolbox.register("mutate", tools.mutESLogNormal, c=1.0, indpb=0.03)
toolbox.register("select", tools.selNSGA2)
toolbox.register("evaluate", wrapper)

# Application de la borne minimale
toolbox.decorate("mate", checkStrategy(MIN_STRATEGY))
toolbox.decorate("mutate", checkStrategy(MIN_STRATEGY))

def launch_nsga2(mu=100, lambda_=200, cxpb=0.6, mutpb=0.3, ngen=1000, display=False, verbose=False):
    
    random.seed()
    
    population = toolbox.population(n=mu)
    paretofront = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    
    ## A completer: évaluation des individus et mise à jour de leur fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    
    if paretofront is not None:
        paretofront.update(population)
    
    print("Pareto Front: "+str(paretofront))

record = stats.compile(population) if stats is not None else {}
logbook.record(gen=0, nevals=len(invalid_ind), **record)
if verbose:
    print(logbook.stream)
    
    population = toolbox.select(population, len(population))
    
    # Begin the generational process
    for gen in range(1, ngen + 1):
        
        
        ## A completer, génération des 'offspring' et sélection de la nouvelle population
        offspring = tools.selTournamentDCD(population, len(population))
        offspring = [toolbox.clone(ind) for ind in offspring]
        
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= cxpb:
                toolbox.mate(ind1, ind2)
            
            if random.random() <= mutpb:
                toolbox.mutate(ind1)
            
            if random.random() <= mutpb:
                toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values
    
        
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        population = toolbox.select(population + offspring, len(population))
        
        # Update the hall of fame with the generated individuals
        if paretofront is not None:
            paretofront.update(offspring)

if display:
    plot_pop_pareto_front(population, paretofront, "Gen: %d"%(gen))
        
        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

return population, logbook, paretofront

if __name__ == '__main__':
    population,logbook,paretofront = launch_nsga2(display=True)
