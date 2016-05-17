"""
A module for recognizing rare behaviors.
"""

# pylint: disable=missing-docstring
# pylint: disable=too-many-locals

import pickle
import os.path
import string
import sys
import yesno

import progress

from ball_keeper import FEATURE_NAMES

import sampling2

from sampling2 import Meta

import math
import csv

from combinations import combinations


class ValueBinner(object):
    """
    A helper class for binning of behavior value.
    """

    def __init__(self, description, minimum, maximum, bin_count):
        self.diff = float(maximum - minimum)
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.bin_count = bin_count
        self.description = description

    def get_description(self):
        """
        returns the description of the associated bin.
        """
        return self.description

    def get_range(self, index):
        """
        returns the range of values (lower, upper)
        covered by the bin with the given index.
        """

        bins = self.bin_count

        # print "index = %d " % (index)
        normalized = (float(index)) / (float(bins))
        # print "min = %f, max = %f" % (self.minimum, self.maximum)
        # print "normalized = %f" % (normalized)
        diff = self.diff
        mini = self.minimum

        lower = (normalized * diff) + mini
        upper = lower + (diff / bins)

        # print "lower = %f, upper = %f" % (lower, upper)

        # print "Enter to continue."
        # raw_input()

        return (lower, upper)

    def bin_value(self, value, verbose=False):
        """
        returns the binning index of the given value.
        """

        if verbose:
            print "binning %f (%f to %f)" % (value, self.minimum, self.maximum)

        diff = self.diff
        mini = self.minimum
        bins = self.bin_count

        if diff == 0:
            return (float(bins)) * 0.5
        else:
            normalized = (float(value) - mini) / diff
            index = int(normalized * bins)

            if normalized >= 1:
                if normalized > 1 and verbose:
                    print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                    print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                    print "BREACHED BOTTOM OF BINS"
                    print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                    print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                index = bins - 1
            elif normalized <= 0:
                if normalized < 0 and verbose:
                    print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                    print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                    print "BREACHED TOP OF BINS"
                    print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                    print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                index = 0

            if verbose:
                print "normalized: %f ; bin: %d" % (normalized, index)

            return index


class RarityTable(object):
    """
    A class for calculating rarity of behaviors.
    """

    def __init__(self, column_names, data, data_size, binners):
        self.column_names = column_names
        self.data_size = float(data_size)
        self.data = data
        self.binners = binners

    def get_bin_count(self):
        binner = self.binners[0]
        return binner.bin_count

    def get_histogram(self, columns):
        """
        returns the histogram for the feature-set corresponding
        to the given column indices.
        """
        return self.data[columns]

    def debug_print_single(self, columns):
        """
        A debugging function for printing the contents of
        a single histogram, for the feature-set corresponding
        to the given column indices.
        """
        single = self.data[columns]
        column_names = self.column_names
        print [column_names[column] for column in list(columns)]

        for key in single:
            value = single[key]
            print "\t %s\n\t-> %f" % (key, value)

    def debug_print_table(self):
        """
        A debugging function for printing the contents of
        the entire rarity table.
        """
        data_items = self.data.items()

        data_items.sort(key=lambda k_v: (len(k_v[0]), k_v[0]))

        for (columns, histogram) in data_items:
            print [FEATURE_NAMES[column] for column in list(columns)]

            histogram_items = histogram.items()
            histogram_items.sort(key=lambda k__: (len(k__[0]), k__[0]))
            #sum = 0.0
            for (key, value) in histogram_items:
                    # sum += value
                print "  %s\t%f" % (key, value)

            # print "\tsum\t-> %f" % (sum)

            show_more = yesno.query("View more?")
            if show_more:
                print "\n"
            else:
                return

    def write_to_csv(self, csv_file):
        """
        A helper function for writing the histograms
        to a CSV file.
        """
        data_items = self.data.items()

        data_items.sort(key=lambda k_v2: (len(k_v2[0]), k_v2[0]))

        csv_writer = csv.writer(csv_file, delimiter=';', quotechar='"')

        column_count = 1

        column_names = self.column_names

        for (columns, histogram) in data_items:

            if len(columns) > column_count:
                column_count = len(columns)
                csv_writer.writerow(['*' * 32] * (column_count + 1))

            header = [column_names[column] for column in columns]
            header.append("count")
            csv_writer.writerow(header)

            histogram_items = histogram.items()
            histogram_items.sort(key=lambda k__1: (len(k__1[0]), k__1[0]))
            for (key, value) in histogram_items:

                #print (key, value)
                def key_with_range(elm):
                    """
                    returns a string containing the given key,
                    and the range of the corresponding bin.
                    """
                    (index, key) = elm
                    #print (index, key)
                    (lower, upper) = self.binners[index].get_range(key)
                    return "%d [%f --- %f]" % (key, lower, upper)

                #row = list(key)
                row = [key_with_range(elm) for elm in zip(columns, list(key))]

                row.append(value)
                csv_writer.writerow(row)

            csv_writer.writerow([])

    def get_count(self, behavior, columns):
        """
        returns the recorded amount of occurrences
        of the binned features of the given behavior.
        Only the featureset corresponding to the given
        columns are regarded.
        """
        # print "behavior: %s" % (list(behavior))

        bin_indices = []

        for (idx, column) in enumerate(columns):
            val = behavior[idx]

            binner = self.binners[column]
            bin_index = binner.bin_value(val, False)
            bin_indices.append(bin_index)

        # print "bin_indices %s" % (bin_indices)

        hist = self.data[columns]

        tupl = tuple(bin_indices)
        # print "tuple %s" % (t)

        count = None

        if tupl in hist:
            count = hist[tupl]

        return count

    def get_probability(self, behavior, columns):
        """
        returns the recorded probability of behaviors
        expressing similar featuresets as the ones
        indicated by the columns of the given behavior.
        """

        # print "behavior: %s" % (list(behavior))

        bin_indices = []

        for (idx, column) in enumerate(columns):
            val = behavior[idx]

            binner = self.binners[column]
            bin_index = binner.bin_value(val, False)
            bin_indices.append(bin_index)

        # print "vals %s" % (vals)
        # print "bins %s" % (bins)

        hist = self.data[columns]

        tupl = tuple(bin_indices)
        # print "tuple %s" % (t)

        if tupl in hist:
            count = hist[tupl]
            probability = (float(count)) / (float(self.data_size))
        else:
            probability = (1.0 / (float(self.data_size + 1)))
        

        return probability

    def get_rarity(self, behavior, columns):
        """
        returns the rarity of the feature-set
        indicated by the columns of the given behavior.
        """
        probability = self.get_probability(behavior, columns)

        rarity = float("inf")

        if probability > 0:
            rarity = -math.log10(probability)

        return rarity


def debug_print_population_size(population):
    """
    A debugging function.
    Used for printing statistics about the
    MultiNEAT population.
    """
    num_species = len(population.Species)
    num_ids = 0

    print "species %d" % (num_species)

    for species in population.Species:
        ids = len(species.Individuals)
        print "---%d" % (ids)
        num_ids += ids

    print "----------"
    print "SUM %d" % num_ids


def build_rarity_table(
        sample_count,
        population_size,
        generations):
    """
    A function for building a rarity table.
    """

    sample_manager = sampling2.SampleManager(population_size, generations)

    column_names = sample_manager.meta.feature_names

    behavior_list = sample_manager.sample_reader(sample_count)

    print "samples loaded, building tables..."

    # collecting list of behaviors in to lists of features

    column_count = len(column_names)

    #bin_count = math.ceil(2.0 * (sample_count ** (1.0/3.0)))
    bin_count = 40
    #bin_count = 20

    min_list = [None] * column_count
    max_list = [None] * column_count

    print "calculating extremes..."

    p_bar = progress.ProgressBar(32, sample_count)

    for behavior in behavior_list:

        p_bar.post()

        for i in range(column_count):

            value = behavior[i]

            if min_list[i] is None or value < min_list[i]:
                min_list[i] = value

            if max_list[i] is None or value > max_list[i]:
                max_list[i] = value

    p_bar.end()

    print "...extremes calculated."

    print "preparing binners..."

    binner_list = [None] * column_count

    for i in range(column_count):

        description = column_names[i]
        mini = min_list[i]
        maxi = max_list[i]

        binner = ValueBinner(description, mini, maxi, bin_count)

        binner_list[i] = binner

    print "building histograms..."

    histograms = {}

    columns_list = [] 
    
    for columns in combinations(range(column_count)):
        if len(columns) <= 3:
            columns_list.append(columns)

    for columns in columns_list:
        histograms[columns] = {}

    for behavior in behavior_list:

        p_bar.post()

        binned_values = tuple(
            binner_list[i].bin_value(value) for (
                i, value) in enumerate(behavior))

        for columns in columns_list:
            histogram = histograms[columns]

            value = tuple(binned_values[column] for column in columns)

            if value in histogram:
                histogram[value] += 1
            else:
                histogram[value] = 1

    p_bar.end()

    print "...histograms built!"

    return RarityTable(column_names, histograms, sample_count, binner_list)


def get_rarity_table_path(
        sample_count,
        population_size,
        generations):

    formatter = string.Formatter()

    dir_name = "rarity_cache"

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    file_name = formatter.format(
        "cache_{0}_{1}_{2}.pkl",
        sample_count,
        population_size,
        generations)

    file_path = os.path.join(dir_name, file_name)

    return file_path


def load_rarity_table(file_path):

    try:
        pkl_file = open(file_path, "rb")

        rarity_table = pickle.load(pkl_file)

        pkl_file.close()

        return rarity_table
    except IOError:
        return None


def save_rarity_table(rarity_table, file_path):

    pkl_file = open(file_path, "wb")

    pickle.dump(rarity_table, pkl_file)

    pkl_file.close()


def get_rarity_table(
        sample_count,
        population_size,
        generations):

    table_path = get_rarity_table_path(
        sample_count, population_size, generations)

    print "Loading rarity table..."

    table = load_rarity_table(table_path)

    if table is None:
        print "...could not find a cached table."
        print "Creating new table..."
        table = build_rarity_table(sample_count, population_size, generations)
        print "...caching table..."
        save_rarity_table(table, table_path)

    print "...table ready!"
    return table


def str2bool(the_string):
    return (str(the_string)).lower() in ("yes", "true", "t", "1")


def save_to_csv(sample_count, population_size, generations, table):
    formatter = string.Formatter()

    dir_name = "histograms"

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    file_name = formatter.format(
        "histogram_{0}_{1}_{2}.csv",
        sample_count,
        population_size,
        generations)

    file_path = os.path.join(dir_name, file_name)

    print "Saving CSV to %s..." % (file_path)

    csv_file = open(file_path, "w+")

    table.write_to_csv(csv_file)

    csv_file.close()

    print "...saved!"


def main(args):
    """
    The main entrypoint.
    """

    from flags import FlagParser, FlagQuerier

    if len(args) <= 1:
        querier = FlagQuerier()

        sample_count = int(querier.sample_count)
        population_size = int(querier.population_size)
        generations = int(querier.generations)

    else:
        parser = FlagParser()
        parser.parse_flags(sys.argv)

        sample_count = parser.sample_count
        population_size = parser.population_size
        generations = parser.generations

    print "Preparing rarity table..."

    table = get_rarity_table(
        sample_count,
        population_size,
        generations)

    make_csv = yesno.query("Do you wish to save the histograms to a CSV-file?")

    if make_csv:
        save_to_csv(sample_count, population_size, generations, table)

if __name__ == "__main__":
    main(sys.argv)
