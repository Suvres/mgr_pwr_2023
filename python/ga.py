"""
    Algorytm Genetyczny

    TODO: Opis GA do inżynierki

"""
from functools import partial
from random import choices, randint, randrange, random
from typing import List, Callable, Tuple
import pandas as pd
import numpy as np
from helper import result_log, save_solution, print_stats

from gnb import simple_gnb

Genome = List[int]
Population = List[Genome]
Columns = np.array

FitnessFunc = Callable[[Genome], int]
PopulateFunc = Callable[[], Population]
SelectionFunc = Callable[[Population, List[float]], Tuple[Genome, Genome]]
PopulationSortedFunc = Callable[[Population, FitnessFunc], Tuple[Population, List[float]]]
CrossoverFunc = Callable[[Genome, Genome], Tuple[Genome, Genome]]
MutationFunc = Callable[[Genome], Genome]

# Generuje genotyp o długości wynoszącej ilość kolumn w tabeli
def generate_genome(columns: Columns) -> Genome:
    return choices([0, 1], k=len(columns))


# Generuje populacje składającą się z wielu genów
def generate_population(size: int, columns: Columns) -> Population:
    return [generate_genome(columns) for _ in range(size)]


# Krzyżowanie 1 punktowe
def single_point_crossover(a: Genome, b: Genome) -> Tuple[Genome, Genome]:
    if len(a) != len(b):
        raise ValueError("Musi być ta sama długość genomów")

    length = len(a)
    if length < 2:
        return a, b

    p = randint(1, length - 1)
    return a[0:p] + b[p:], b[0:p] + a[p:]


# Mutacja genomów dla "n" indeksów
def mutation(genome: Genome, num: int = 1, probability: float = 0.5) -> Genome:
    for _ in range(num):
        index = randrange(len(genome))
        genome[index] = genome[index] if random() > probability else abs(genome[index] - 1)

    return genome


# Zwracanie wyniku jako kolumn dla tabeli
def genome_to_columns(genome: Genome, columns: Columns) -> Columns:
    result = []

    for i in range(len(columns)):
        if genome[i] == 1:
            result += [columns[i]]

    return result


# Zwraca parę obiektów dla wag wygenerowanych za pomocą metody fitness
def selection_pairs(population: Population, weights: List[float]) -> Population:
    return choices(
        population=population,
        weights=weights,
        k=2
    )


def calculate_simple(genome: Genome, columns: Columns, dataset: pd.DataFrame,
                     dataset_test: pd.DataFrame):
    logical_genome = [i == 1 for i in genome]

    tmp_columns = np.append(columns[logical_genome], "Label")
    tmp_dataset = dataset[tmp_columns]
    tmp_dataset_test = dataset_test[tmp_columns]

    gnb_result = simple_gnb(dataset=tmp_dataset, dataset_test=tmp_dataset_test)
    result_log(result=gnb_result, columns=tmp_columns)
    return gnb_result 


# Fitness
def fitness(genome: Genome, columns: Columns, dataset: pd.DataFrame, dataset_test: pd.DataFrame) -> int:
    if len(genome) != len(columns):
        raise ValueError("Musi być ta sama długość")

    calculate_result = calculate_simple(genome=genome, columns=columns, dataset=dataset, dataset_test=dataset_test)
    return calculate_result.loc[0, "accuracy"]


def population_sorted(population: Population, fitness_func: FitnessFunc) -> Tuple[Population, List[float]]:
    weights = []
    for i in range(len(population)):
        weights.append(fitness_func(population[i]))

    for i in range(len(weights)):
        for j in range(len(weights)):
            if weights[j] < weights[i]:
                weights[j], weights[i] = weights[i], weights[j]
                population[j], population[i] = population[i], population[j]

    return population, weights


def run_evolution(
        populate_func: PopulateFunc,
        fitness_func: FitnessFunc,
        selection_func: SelectionFunc = selection_pairs,
        crossover_fun: CrossoverFunc = single_point_crossover,
        mutation_fun: MutationFunc = mutation,
        population_sorted_fun: PopulationSortedFunc = population_sorted,
        generation_limit: int = 100,
        limit: float = 0.9
) -> Tuple[Population, int]:
    population = populate_func()

    for i in range(generation_limit):
        print("\n\n======= Populacja: {0} ========\n\n".format(i+1))
        population, weights = population_sorted_fun(population, fitness_func)

        if weights[0] > limit:
            break

        next_generation = population[0:2]

        for j in range(int(len(population) / 2) - 1):
            parents = selection_func(population, weights)
            offspring_a, offspring_b = crossover_fun(parents[0], parents[1])

            offspring_a = mutation_fun(offspring_a)
            offspring_b = mutation_fun(offspring_b)

            next_generation += [offspring_a, offspring_b]

        population = next_generation

    population, weights = population_sorted_fun(population, fitness_func)

    return population, i


def ga_simple(dataset1, dataset2):
    columns = dataset1.columns.tolist()
    columns = np.array(columns[:-1])
    # TODO do zmiany ilość iteracji na 1000
    generations_limit = 1000
    
    run_evolution(
        populate_func=partial(
            generate_population, size=10, columns=columns
        ),
        fitness_func=partial(
            fitness, columns=columns, dataset=dataset1, dataset_test=dataset2
        ),
        generation_limit=generations_limit,
    )
    columns = np.append(columns, "Label")

    print(columns)
    return columns
