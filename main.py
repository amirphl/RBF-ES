import sys
import csv
import deap.algorithms as al
from deap import tools
from es_utils import *
from network_utils import get_precision, generate_W_matrix, generate_G_matrix


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


def prepare_data(number_of_lines, path, shuffle=0):
    with open(path, 'r') as f:
        train_data = list(csv.reader(f, delimiter=","))
    train_data = train_data[1::]

    if number_of_lines > len(train_data):
        print("Not enough lines in data file to read, instead used", len(train_data), "lines.")
        number_of_lines = len(train_data)

    if shuffle == 1:
        train_data = random.choices(train_data, k=number_of_lines)
        print("data shuffled...")
    else:
        train_data = train_data[0:number_of_lines]
    train_data = np.array(train_data, dtype=np.float32)
    r, c = train_data.shape
    y = train_data[:, c - 1]
    X = train_data[:, 0:c - 1]
    L, n = X.shape

    scales = []
    for i in range(n):
        # X[:, i] = X[:, i] / np.linalg.norm(X[:, i])
        scales.append(X[:, i].max())
        X[:, i] = X[:, i] / X[:, i].max()

    return X, y, scales


def prepare_data_for_multi_class_classification(y):
    l = y.shape[0]
    c = int(np.max(y))
    y_prime = np.zeros(shape=(l, c), dtype=np.float32)
    for i in range(l):
        y_prime[i, int(y[i]) - 1] = 1.0
    return y_prime


def store_network_weights(X, y, V, gama):
    G = generate_G_matrix(X, V, gama)
    W = generate_W_matrix(G, y)

    print("W:")
    print(W)
    print("V:")
    print(V)
    print("gama:")
    print(gama)

    top = sys.argv[4]
    np.save(file="outputs/weights" + top + ".npy", arr=W)
    np.save(file="outputs/V" + top + ".npy", arr=V)
    np.save(file="outputs/gama" + top + ".npy", arr=gama)


if __name__ == '__main__':
    path_to_train_data = str(sys.argv[1])
    lines = int(sys.argv[2])  # number of lines to be read from train data
    m = int(sys.argv[3])  # number of V vectors
    type_of_problem = int(sys.argv[4])
    m_ngen = int(sys.argv[5])  # number of generations
    m_shuffle = int(sys.argv[6])  # shuffle = 1 or not

    m_X, m_y, m_scales = prepare_data(lines, path_to_train_data, shuffle=m_shuffle)
    m_l, m_n = m_X.shape
    m_min_value = m_X.min()
    m_max_value = m_X.max()
    m_c = 0

    if type_of_problem == 2:
        m_y = prepare_data_for_multi_class_classification(m_y)
        _, m_c = m_y.shape

    toolbox = initialize(m_X, m_y, n=m_n, m=m, mu=10, min_value=m_min_value, max_value=m_max_value)
    m_pop, m_logbook, m_hof = train(toolbox, mu=10, ngen=m_ngen)

    m_ind = m_hof.items[0]
    m_V = m_ind.V.reshape(m, m_n)
    m_gama = m_ind.gama

    print("Error on train data set:", evaluate_parameters(m_V, m_gama, m_X, m_y))
    if type_of_problem == 1 or type_of_problem == 2:
        print("Precision on train data set:", get_precision(m_X, m_y, m_V, m_gama))

    store_network_weights(m_X, m_y, m_V, m_gama)
