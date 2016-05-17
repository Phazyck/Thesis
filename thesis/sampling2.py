# pylint: disable=line-too-long
# pylint: disable=missing-docstring
# pylint: disable=too-few-public-methods

import os
import pickle
import string
import sys
import sampling1
import config
import yesno
from array import array
from ball_keeper import FEATURE_NAMES


class Meta(object):

    def __init__(self, feature_names):
        self.sample_count = 0
        self.feature_names = feature_names
        self.feature_count = len(feature_names)


class SampleReader(object):

    def __init__(self, samples, amount):
        self.samples = samples
        self.amount = amount
        self.data_file = None
        self.sample_index = 0

    def __iter__(self):
        if self.samples.busy:
            raise Exception("Cannot read samples, source file is in use.")

        # r = read, position at the start | b = binary mode
        mode = 'rb'
        self.samples.busy = True
        self.data_file = open(self.samples.data_path, mode)
        self.sample_index = 0

        return self

    def __next__(self):
        if self.sample_index >= self.amount:
            self.data_file.close()
            self.samples.busy = False
            raise StopIteration

        array_length = self.samples.feature_count()

        sample = array('d')

        sample.fromfile(self.data_file, array_length)

        self.sample_index += 1

        return sample

    def next(self):
        return self.__next__()


class SampleWriter(object):

    def __init__(self, samples):
        self.samples = samples
        self.data_file = None

    def open(self):
        if self.samples.busy:
            raise Exception("Trying to write while reading!")

        self.samples.busy = True

        # a = write, create, position at end | b = binary mode
        mode = "ab"
        self.data_file = open(self.samples.data_path, mode)

    def write_one(self, sample):
        self.samples.meta.sample_count += 1

        float_array = array('d', sample)

        float_array.tofile(self.data_file)

    def write_many(self, samples):
        for sample in samples:
            self.write_one(sample)

    def close(self):
        self.data_file.close()

        mode = 'wb'
        meta_file = open(self.samples.meta_path, mode)
        pickle.dump(self.samples.meta, meta_file)
        meta_file.close()

        self.samples.busy = False


class SampleManager(object):

    def __init__(self, population_size, generations, rebuild=False):

        self.population_size = population_size
        self.generations = generations

        feature_names = FEATURE_NAMES

        # prepare file paths
        formatter = string.Formatter()

        dir_name = "samples"

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        meta_name = formatter.format(
            "meta_{0}_{1}.pkl",
            population_size,
            generations
        )

        data_name = formatter.format(
            "data_{0}_{1}.bin",
            population_size,
            generations
        )

        meta_path = os.path.join(dir_name, meta_name)
        data_path = os.path.join(dir_name, data_name)

        self.meta_path = meta_path
        self.data_path = data_path

        # ensure that files exist

        if (not (os.path.exists(meta_path) and os.path.exists(data_path))) or rebuild:

            # w = write, create, truncate, position at start | b = binary mode
            mode = 'wb'

            # create an initial meta file
            meta_file = open(meta_path, mode)
            pickle.dump(Meta(feature_names), meta_file)
            meta_file.close()

            # create an empty data file
            data_file = open(data_path, mode)
            data_file.close()

        # load meta information

        # r = read, position at start | b = binary mode
        mode = 'rb'
        meta_file = open(meta_path, mode)
        self.meta = pickle.load(meta_file)
        meta_file.close()

        self.busy = False

    def sample_count(self):
        return self.meta.sample_count

    def sample_reader(self, amount):

        missing = amount - self.meta.sample_count

        if missing > 0:

            print "Requested %d samples." % amount
            print "The cache only contains %d samples." % self.meta.sample_count
            question = "Would you like to extend the cache with %d samples?" % missing
            extend = yesno.query(question)

            if extend:
                additional_samples = sampling1.generate_samples(
                    missing, self.population_size, self.generations)

                writer = self.sample_writer()
                writer.open()

                for sample in additional_samples:
                    writer.write_one(sample)
                writer.close()

            else:
                print "Fine.. If you're gonna be like that.."
                print "[ terminated ]"

        return SampleReader(self, amount)

    def sample_writer(self):

        return SampleWriter(self)

    def feature_names(self):
        return self.meta.feature_names

    def feature_count(self):
        return self.meta.feature_count


def port_csv(population_size, generations):

    (_, _, sample_file) = sampling1.load_samples(population_size, generations)

    sample_count = 0

    for _ in sample_file:
        sample_count += 1

    (_, sample_reader, sample_file) = sampling1.load_samples(
        population_size, generations)

    sample_manager = SampleManager(population_size, generations, rebuild=True)

    writer = sample_manager.sample_writer()

    writer.open()

    ported = 0

    for sample in sample_reader:
        values = [float(value) for value in sample]
        writer.write_one(values)

        ported += 1

        if ported % 10000 == 0:
            pct = 100.0 * (float(ported) / float(sample_count))
            print '  %d of %d samples ported. (%5.1f%%)\r' % (ported, sample_count, pct),
            sys.stdout.flush()

    print '  %d of %d samples read. (%5.1f%%)' % (ported, sample_count, pct)

    writer.close()

    sample_file.close()

    print "Porting finished!"


def _main(args):
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
        
    sample_manager = SampleManager(population_size, generations)
    behavior_list = sample_manager.sample_reader(sample_count)
    
    behavior_count = 0
    for _ in behavior_list:
        behavior_count += 1
        
    print "%d samples ready!" % behavior_count
    

if __name__ == '__main__':
    _main(sys.argv)


    #port_csv(64, 32)

    """
    manager = SampleManager(10, 10)

    target = 1000000

    reader = manager.sample_reader(target)

    for sample in reader:
        print sample

    """

    """
    feature_names = ['a', 'b', 'c']
    s =  SampleManager(feature_names, 10, 10)

    target = 1000001

    c = s.sample_count()

    print "sample_count = %d" % (c)

    diff = target - c

    if diff > 0:

        print "need %d more samples" % (diff)

        raw_input()

        writer = s.sample_writer()

        writer.open()

        for i in range(diff):



            if i % 10000 == 0:
                print "%d samples written." % i

            writer.write_one([i,i,i])

        writer.close()

    raw_input()

    reader = s.sample_reader(target)

    def read_samples():

        read = 0
        for sample in reader:
            read += 1
            if read % 10000 == 0:
                pct = 100.0 * (float(read) / float(target))
                print '  %d samples read. (%5.1f%%)\r' % (read, pct),
                sys.stdout.flush()

        print '  %d samples read. (%5.1f%%)' % (read, pct)

        print "\ndone!"
        raw_input()

    read_samples()
    read_samples()
    """
