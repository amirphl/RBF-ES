from es_utils import *
from network_utils import get_precision, generate_W_matrix, generate_G_matrix, get_train_error
import sys
import csv
import random
import deap.algorithms as al
from deap import tools

TYPE_OF_PROBLEM = 0  # 0 : regression , 1 : classification with 2 class , 2 : classification with more than 2 class


def generic_train(toolbox, pop_size=10, CXPB=0.5, MUTPB=0.2, NGEN=20):
    """
    dont use this function...
    :param toolbox:
    :param pop_size:
    :param CXPB:
    :param MUTPB:
    :param NGEN:
    :return:
    """
    population = toolbox.population_guess()

    # Evaluate the entire population
    fitnesses = map(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    for g in range(NGEN):
        # Select the next generation individuals
        offspring = toolbox.select(population, pop_size)
        # Clone the selected individuals
        offspring = map(toolbox.clone, offspring)
        offspring = list(offspring)

        # Apply crossover on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Apply mutation on the offspring
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        population += offspring
        population = toolbox.select(population, pop_size)
        population = population[:]
    best_ind = None
    best_ind_fitness = 1000000

    for ind in population:
        if toolbox.evaluate(ind)[0] < best_ind_fitness:
            best_ind = ind
    return best_ind


def train(toolbox, mu=10, m_lambda=100, cxpb=0.6, mutpb=0.3, ngen=10):
    pop = toolbox.population_guess()
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, logbook = al.eaMuCommaLambda(pop, toolbox, mu=mu, lambda_=m_lambda,
                                      cxpb=cxpb, mutpb=mutpb, ngen=ngen, stats=stats, halloffame=hof)

    return pop, logbook, hof


def prepare_data(number_of_lines, path):
    with open(path, 'r') as f:
        train_data = list(csv.reader(f, delimiter=","))

    if number_of_lines > len(train_data):
        print("Not enough lines in data file, used", len(train_data), "lines instead.")
        number_of_lines = len(train_data)

    train_data = np.array(train_data[0:number_of_lines], dtype=np.float32)
    r, c = train_data.shape
    y = train_data[:, c - 1]
    X = train_data[:, 0:c - 1]
    L, n = X.shape

    scales = []
    for i in range(n):
        # X[:, i] = X[:, i] / np.linalg.norm(X[:, i])
        my_max = X[:, i].max()
        X[:, i] = X[:, i] / my_max
        scales.append(my_max)
    return X, y, L, n, scales


def store_network_weights(X, y, V, gama):
    G = generate_G_matrix(X, V, gama)
    W = generate_W_matrix(G, y)

    print("W:")
    print(W)
    print("V:")
    print(V)
    print("gama:")
    print(gama)

    np.save(file="weights.npy", arr=W)
    np.save(file="V.npy", arr=V)
    np.save(file="gama.npy", arr=gama)


if __name__ == '__main__':
    path_to_train_data = str(sys.argv[1])
    lines = int(sys.argv[2])  # number of lines to be read from train data
    m = int(sys.argv[3])  # number of V vectors
    TYPE_OF_PROBLEM = int(sys.argv[4])

    X, y, L, n, scales = prepare_data(lines, path_to_train_data)

    toolbox = initialize(X, y, n=n, m=m, mu=10)

    pop, logbook, hof = train(toolbox, mu=10)

    ind = hof.items[0]
    V = ind.V.reshape(m, n)
    gama = ind.gama

    print("Error on train data set:", evaluate_parameters(V, gama, X, y))
    if TYPE_OF_PROBLEM == 1 or TYPE_OF_PROBLEM == 2:
        print("Precision on train data set:", get_precision(X, y, V, gama))

    store_network_weights(X, y, V, gama)
