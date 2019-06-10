import numpy as np
from deap import base, creator, tools
import json
import random
from network_utils import evaluate_parameters
from itertools import repeat
from collections import Sequence
from operator import attrgetter


class MyIndividual:
    V = np.eye(1)
    gama = np.eye(1)


def initialize(X, y, minimize=True, path_to_json_initializer="guess.json", n=2, m=5, mu=0, sigma=1, indpb=0.1,
               tournsize=3):
    if minimize:
        creator.create("Fitness", base.Fitness, weights=(-1.0,))
    else:
        creator.create("Fitness", base.Fitness, weights=(1.0,))

    creator.create("Individual", MyIndividual, fitness=creator.Fitness)

    toolbox = base.Toolbox()

    toolbox.register("individual_guess", initIndividual, creator.Individual)
    toolbox.register("population_guess", initPopulation, list, toolbox.individual_guess, path_to_json_initializer, n, m)

    toolbox.register("mate", cxTwoPoint)  # TODO choose better crossover method for ES
    toolbox.register("mutate", mutGaussian, mu, sigma, indpb)  # TODO choose better mutation method for ES
    toolbox.register("select", selTournament, tournsize=tournsize)  # TODO choose better selection method for ES
    toolbox.register("evaluate", evaluate, n, m, X, y)

    return toolbox


def initIndividual(icls, content, n, m):
    V_vector = np.ndarray(shape=(1, n * m))
    gama_vector = np.ndarray(shape=(m, 1))
    # TODO fill v and gama from file
    V_vector.fill(0.5)
    gama_vector.fill(0.5)
    my_individual = MyIndividual
    my_individual.gama = gama_vector
    my_individual.V = V_vector
    return icls(my_individual)


def initPopulation(pcls, ind_init_guess, filename, n, m):
    with open(filename, "r") as pop_file:
        contents = json.load(pop_file)
    return pcls(ind_init_guess(c, n, m) for c in contents)


def evaluate(individual, n, m, X, y):
    V_matrix = individual.V.reshape((m, n))
    gama_vector = individual.gama
    fitness = evaluate_parameters(V_matrix, gama_vector, X, y)
    return fitness


def mutGaussian(mu, sigma, indpb, individual):
    """This function applies a gaussian mutation of mean *mu* and standard
    deviation *sigma* on the input individual. This mutation expects a
    numpy individual composed of real valued attributes.
    The *indpb* argument is the probability of each attribute to be mutated.

    :param individual: Individual to be mutated.
    :param mu: Mean or :term:`python:sequence` of means for the
               gaussian addition mutation.
    :param sigma: Standard deviation or :term:`python:sequence` of
                  standard deviations for the gaussian addition mutation.
    :param indpb: Independent probability for each attribute to be mutated.
    :returns: A tuple of one individual.

    This function uses the :func:`~random.random` and :func:`~random.gauss`
    functions from the python base :mod:`random` module.

    This function is very similar to deap.tools.mutGaussian
    """
    V_vector = individual.V
    gama_vector = individual.gama
    vector_mutation(V_vector, mu, sigma, indpb)
    vector_mutation(gama_vector, mu, sigma, indpb)
    return individual,


def vector_mutation(individual, mu, sigma, indpb):
    _, size = individual.shape
    if not isinstance(mu, Sequence):
        mu = repeat(mu, size)
    elif len(mu) < size:
        raise IndexError("mu must be at least the size of individual: %d < %d" % (len(mu), size))
    if not isinstance(sigma, Sequence):
        sigma = repeat(sigma, size)
    elif len(sigma) < size:
        raise IndexError("sigma must be at least the size of individual: %d < %d" % (len(sigma), size))

    for i, m, s in zip(range(size), mu, sigma):
        if random.random() < indpb:
            individual[0, i] += random.gauss(m, s)


def cxTwoPoint(ind1, ind2):
    """Executes a two-point crossover on the input numpy
    individuals. The two individuals are modified in place and both keep
    their original length.

    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :returns: A tuple of two individuals.

    This function uses the :func:`~random.randint` function from the Python
    base :mod:`random` module.

    This function is very similar to deap.tools.cxTwoPoint
    """
    V_vector_1 = ind1.V
    gama_vector_1 = ind1.gama
    V_vector_2 = ind2.V
    gama_vector_2 = ind2.gama

    mated_V_vector_1, mated_V_vector_2 = vector_crossover(V_vector_1, V_vector_2)
    mated_gama_vector_1, mated_gama_vector_2 = vector_crossover(gama_vector_1, gama_vector_2)

    new_individual_1 = MyIndividual
    new_individual_2 = MyIndividual

    new_individual_1.V = mated_V_vector_1
    new_individual_1.gama = mated_gama_vector_1

    new_individual_2.V = mated_V_vector_2
    new_individual_2.gama = mated_gama_vector_2

    return new_individual_1, new_individual_2


def vector_crossover(ind1, ind2):
    _, s1 = ind1.shape
    _, s2 = ind2.shape
    size = min(s1, s2)
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:  # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1 = ind1.copy()
    ind2 = ind2.copy()
    ind1[0, cxpoint1:cxpoint2], ind2[0, cxpoint1:cxpoint2] = ind2[0, cxpoint1:cxpoint2].copy(), ind1[0,
                                                                                                cxpoint1:cxpoint2].copy()

    return ind1, ind2


def selTournament(individuals, k, tournsize, fit_attr="fitness"):
    """Select the best individual among *tournsize* randomly chosen
    individuals, *k* times. The list returned contains
    references to the input *individuals*.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :param tournsize: The number of individuals participating in each tournament.
    :param fit_attr: The attribute of individuals to use as selection criterion
    :returns: A list of selected individuals.

    This function uses the :func:`~random.choice` function from the python base
    :mod:`random` module.

    This function is exactly as same as deap.tools.selTournament
    """
    chosen = []
    for i in range(k):
        aspirants = [random.choice(individuals) for _ in range(tournsize)]
        chosen.append(max(aspirants, key=attrgetter(fit_attr)))
    return chosen