"""
module for parallelize behavior sampling of randomly generated neural networks.
"""

# pylint: disable=missing-docstring
# pylint: disable=too-many-locals
# pylint: disable=pointless-string-statement

from multiprocessing import Process, Queue, cpu_count

import time
import MultiNEAT as NEAT
import ball_keeper
import sys
import csv
import string
import os
import config
from draw_net import DrawGene

def make_genomes(population_size, generations):
    """
    A function for randomly generating and evolving
    a population of genomes.
    """

    parameters = NEAT.Parameters()
    parameters.PopulationSize = population_size

    seed = config.get_neat_genome_seed(parameters)
    randomize_weights = True
    randomization_range = 1.0
    rng_seed = 10

    population = NEAT.Population(
        seed, parameters,
        randomize_weights, randomization_range, rng_seed)

    population.RNG.Seed(rng_seed)

    import random as rng

    for _ in range(generations):
        genome_list = NEAT.GetGenomeList(population)
        genome_count = len(genome_list)
        fitness_list = [rng.random() for _ in xrange(genome_count)]
        NEAT.ZipFitness(genome_list, fitness_list)
        population.Epoch()

    genome_list = NEAT.GetGenomeList(population)
    
    return genome_list

def inspect(population_size, generations):
    genomes = make_genomes(population_size, generations)
    
    for genome in genomes:
        DrawGene(genome)
        raw_input()
   
_SAMPLES_PER_GENE = 10
    
def sample_task(
        queue,
        population_size,
        generations):
    """
    A sampling task to be performed in parallel by a separate process.

    Generated samples are put in the given queue.

    The population_size defines how big the population
    of neural networks should be.

    The generations defines how many generations the population
    should be evolved over.
    """

    game = ball_keeper.init()

    while True:
        genome_list = make_genomes(population_size, generations)

        sample_list = []

        for genome in genome_list:
            for _ in range(_SAMPLES_PER_GENE):
                sample = ball_keeper.evaluate_genome(game, genome, use_random=True).all_features()
                sample_list.append(sample)

        queue.put(sample_list)


def print_progress2(time_a, numerator, denominator):
    """
    A utility function for pretty-printing
    the progress of the sampling process.
    """

    fmt = "%" + str(len(str(denominator))) + "d"

    pct = 100.0 * (float(numerator)) / ((float(denominator)))

    info = (fmt + " of") % (numerator)
    info += " %d samples " % (denominator)
    info += "(%7.3f%%) " % (pct)

    time_b = time.time()

    time_d = time_b - time_a

    def time_info(the_time):
        """
        Internal function for pretty-printing time values.
        prefix and suffix are used to specify text that
        should be printed before an after the time, respectively.
        """
        time_m, time_s = divmod(the_time, 60)
        time_h, time_m = divmod(time_m, 60)

        return "%2dh%2dm%5.2fs" % (int(time_h), int(time_m), time_s)

    total_time = time_d * (100.0 / pct)
    time_left = total_time - time_d

    info += "estimated time left: "

    info += time_info(time_left)

    print ("%s\r" % info),

    sys.stdout.flush()


def print_progress(time_a, numerator, denominator):
    """
    A utility function for pretty-printing
    the progress of the sampling process.
    """

    info = "\n" + ("-" * 64) + "\n"

    fmt = "%" + str(len(str(denominator))) + "d"

    pct = 100.0 * (float(numerator)) / ((float(denominator)))

    info += "\n"
    info += " [ SAMPLE INFO ] \n"
    info += ("  " + fmt + " of\n") % (numerator)
    info += "  %d samples\n" % (denominator)
    info += "\n"
    info += "  %7.3f%% done\n" % (pct)

    time_b = time.time()

    time_d = time_b - time_a

    def time_info(the_time, prefix=None, suffix=None):
        """
        Internal function for pretty-printing time values.
        prefix and suffix are used to specify text that
        should be printed before an after the time, respectively.
        """

        info = ""

        time_m, time_s = divmod(the_time, 60)
        time_h, time_m = divmod(time_m, 60)

        if prefix:
            info += prefix + " "

        if time_h > 0:
            noun = "hour" if time_h < 2 else "hours"
            info += "%d %s," % (int(time_h), noun)

        if time_m > 0 or time_h > 0:
            noun = "minute" if time_m < 2 else "minutes"
            info += "%d %s and " % (int(time_m), noun)

        info += "%5.2f seconds" % (time_s)

        if suffix:
            info += " " + suffix

        info += "\n"

        return info

    total_time = time_d * (100.0 / pct)
    time_left = total_time - time_d

    prefix_d = "          elapsed:"
    prefix_t = "  estimated total:"
    prefix_l = "   estimated left:"

    info += "\n"
    info += " [ TIME INFO ]" + "\n"

    info += time_info(time_d, prefix=prefix_d)
    info += time_info(total_time, prefix=prefix_t)
    info += time_info(time_left, prefix=prefix_l)

    print info


def load_samples(population_size, generations):

    formatter = string.Formatter()

    dir_name = "samples"

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    file_name = formatter.format(
        "samples_{0}_{1}.csv",
        population_size,
        generations)

    file_path = os.path.join(dir_name, file_name)

    try:
        exists = os.path.exists(file_path)

        if not exists:
            sample_file = open(file_path, 'w+')
            sample_writer = csv.writer(
                sample_file, delimiter=';', quotechar='"')
            sample_writer.writerow(ball_keeper.FEATURE_NAMES)
            sample_file.close()

        sample_file = open(file_path, 'r+')
        sample_reader = csv.reader(sample_file, delimiter=';', quotechar='"')

        headers = sample_reader.next()

        return(headers, sample_reader, sample_file)
    except IOError:
        raise


def generate_samples(
        sample_count,
        population_size,
        generations,
        process_count=0):

    time_a = time.time()

    if process_count <= 0:
        process_count = cpu_count()

    queue = Queue()

    process_list = []

    for _ in range(process_count):
        process = Process(
            target=sample_task,
            args=(
                queue,
                population_size,
                generations))
        process_list.append(process)
        process.start()

    sample_list = []

    while True:
        sample_list += queue.get()

        if len(sample_list) > sample_count:
            sample_list = sample_list[:sample_count]

        numerator = len(sample_list)
        denominator = sample_count

        print_progress2(time_a, numerator, denominator)

        if numerator == denominator:
            break

    print "\ndone!"

    for process in process_list:
        process.terminate()
        print "terminated process",
        print process

    return sample_list


def get_samples(sample_count, population_size, generations, process_count=0):
    """
    Public function for issuing a batched sampling
    of randomly generated neural networks.
    """

    (_, _, sample_file) = load_samples(population_size, generations)

    cached_sample_count = 0

    for _ in sample_file:
        cached_sample_count += 1

    print "cached samples = %d" % (cached_sample_count)

    """
    sample_list = []

    t0 = time.clock()
    td = 0.0
    print "reading existing samples from file..."

    interval = 0.2

    for (index, sample) in enumerate(sample_reader):
        if len(sample_list) >= sample_count:
            break

        t1 = time.clock()
        td += t1-t0
        t0 = t1

        if td > interval:
            print "%d samples read" % (index)
            td -= interval

        sample_list.append(sample)

    print "...%d samples read!" % (len(sample_list))
    """
    missing = sample_count - cached_sample_count

    if missing > 0:
        print "Cache does not contain enough samples."
        print "Adding %d additional samples." % (missing)

        additional_samples = generate_samples(
            missing, population_size, generations, process_count)

        sample_writer = csv.writer(sample_file, delimiter=';', quotechar='"')

        for sample in additional_samples:
            sample_writer.writerow(sample)

    sample_file.close()

    (headers, sample_reader, sample_file) = load_samples(
        population_size, generations)

    sample_list = []

    for _ in range(sample_count):
        row = sample_reader.next()
        sample = [float(value) for value in row]
        sample_list.append(sample)

    sample_file.close()

    return (headers, sample_list)


def main(args):
    """
    The main entrypoint.
    """

    from flags import FlagQuerier

    def get_flag(flags, default, cast):
        """
        A utility function for parsing flags
        from a list of arguments.
        """
        for flag in flags:
            try:
                idx = args.index(flag)
                return cast(sys.argv[idx + 1])
            except ValueError:
                # print '"%s" not found, carrying on...' % (flag)
                continue

        return cast(default)

    if len(args) <= 1:
        querier = FlagQuerier()

        process_count = 0
        sample_count = int(querier.sample_count)
        population_size = int(querier.population_size)
        generations = int(querier.generations)

    else:
        process_count = get_flag(["-pc", "--process_count"], 0, int)
        sample_count = get_flag(["-sc", "--sample_count"], 5000, int)
        population_size = get_flag(["-ps", "--population_size"], 64, int)
        generations = get_flag(["-gs", "--generations"], 32, int)

    (_, samples) = get_samples(
        sample_count,
        population_size,
        generations,
        process_count=process_count)

    samples_gotten = sum(1 for _ in samples)

    print "Got %d samples!" % (samples_gotten)

if __name__ == '__main__':
    main(sys.argv)
