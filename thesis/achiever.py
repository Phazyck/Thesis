#!/usr/bin/python3
# -*- coding: utf-8 -*-

# pylint: disable=line-too-long
# pylint: disable=missing-docstring
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=too-few-public-methods
# pylint: disable=too-many-statements

#import cPickle
import os
import sys
#import time
import random
import numpy as numpy
import pickle as pickle
import MultiNEAT as NEAT
#import copy
#import itertools
import ball_keeper
#import networkx as nx

import rarity_recognition
import yesno
from rarity_recognition import RarityTable, ValueBinner
from combinations import combinations
import nbest
from nbest import NBest
import thread
import config

from sampling2 import Meta

# helper function to calculate the novelty of 'behavior' given a list of
# 'behaviors'
def calc_novelty(behavior, behaviors):
    behavior = numpy.array(behavior)

    behaviors = numpy.array(behaviors)

    behaviors -= behavior
    behaviors *= behaviors
    behaviors = sorted(behaviors.sum(1))

    return sum(behaviors[:25]) + 0.00001


class RarityStats(object):

    def __init__(
            self,
            genome, behavior, salient, salient_bin_indices,
            generation, generation_id,
            features,
            support, probability, rarity
    ):

        self.genome = genome
        self.behavior = behavior
        self.salient = salient
        self.salient_bin_indices = salient_bin_indices
        self.generation = generation
        self.generation_id = generation_id
        self.features = features
        self.support = support
        self.probability = probability
        self.rarity = rarity

    def get_ranking(self):
        return self.rarity
        
    def get_value(self):
        return self.behavior

def make_rarity_stats(genome,
                      behavior,
                      feature_indices,
                      generation,
                      generation_id,
                      rarity_table,
                      threshhold_control):

    r_genome = NEAT.Genome(genome)
    r_behavior = behavior
    r_generation = generation
    r_generation_id = generation_id

    import math

    def log(value):
        return math.log(value, 2)

    feature_combinations = combinations(feature_indices)

    bin_count = rarity_table.get_bin_count()

    k = 1

    best_k = None
    best_features = None
    best_support = None
    lowest_probability = None
    best_rarity = None
    best_selected = None

    rarity_threshhold = None
    
    for features in feature_combinations:
    
        features_len = len(features)
        if features_len > k:
            if features_len > best_k + 1:
                break
            else:
                k = features_len
                assumed_probability = lowest_probability / bin_count
                assumed_rarity = -log(assumed_probability)
                best_rarity = -log(lowest_probability)
                assumed_increase = assumed_rarity - best_rarity
                rarity_threshhold = best_rarity + \
                    (threshhold_control * assumed_increase)

        selected = []
        for feature in features:
            idx = feature_indices.index(feature)
            value = behavior[idx]
            selected.append(value)
        selected = tuple(selected)
        
        bin_indices = []
        
        for feature in features:
            idx = filter.feature_indices[feature]
            val = behavior[idx]
            binner = rarity_table.binners[feature]
            bin_index = binner.bin_value(val, False)
            bin_indices.append(bin_index)
            
        if features in rarity_table.data:
            histogram = rarity_table.data[features]
        else:
            break
        
        
        bin_indices = tuple(bin_indices)
        
        if bin_indices in histogram:
            support = histogram[bin_indices]
        else:
            support = None
            
        data_size = float(rarity_table.data_size)
        
        if support is None:
            probability = 1.0 / (data_size + 1.0)
        else:
            probability = float(support) / data_size
        
        rarity = -log(probability)

        if ((best_rarity is None) or (rarity > best_rarity)) and (
                (rarity_threshhold is None) or (rarity > rarity_threshhold)):

            best_k = k
            best_features = [features]
            best_support = [support]
            best_selected = [selected]
            lowest_probability = probability
            best_rarity = rarity
            
        elif ((best_rarity != None) and (rarity == best_rarity)) and (
                (rarity_threshhold is None) or (rarity > rarity_threshhold)):
            best_k = k
            best_features.append(features)
            best_support.append(support)
            best_selected.append(selected)
            lowest_probability = probability
            best_rarity = rarity

    r_probability = lowest_probability
    r_rarity = best_rarity

    stats = []
    for i in range(len(best_features)):
        r_features = best_features[i]
        r_support = best_support[i]
        r_salient = best_selected[i]
        
        r_salient_bin_indices = []
        
        for feature in r_features:
            idx = filter.feature_indices[feature]
            binr = rarity_table.binners[feature]
            desc = rarity_table.column_names[feature]
            val = behavior[idx]
            bin_index = binr.bin_value(val)
            r_salient_bin_indices.append((feature,bin_index))
                
        r_salient_bin_indices = tuple(r_salient_bin_indices)
                
        r_stats = RarityStats(
            r_genome,
            r_behavior,
            r_salient,
            r_salient_bin_indices,
            r_generation,
            r_generation_id,
            r_features,
            r_support,
            r_probability,
            r_rarity)
        stats.append(r_stats)

    return stats

subset = [
    # 0
    "total_frames_passed",
    
    # 1
    "total_collisions_ball_player",
    
    # 2
    "total_collisions_ball_wall",
    
    # 3
    "total_player_jumps",
    
    # 4
    "total_travel_distance_player",
    
    # 5
    "total_travel_distance_ball",

    # 6
    "average_x_position_player",
    
    # 7
    "average_x_position_ball",
    
    # 8
    "average_y_position_ball",
    
    # 9
    "average_distance_ball_player",

    # 10
    "max_ball_velocity",
    
    # 11
    "max_ball_y_position",
    
    # 12
    "max_distance_player_ball",

    # 13
    "final_distance_ball_player"
]

class SubsetFilter(object):
    def __init__(self, subset):
        features = ball_keeper.FEATURE_NAMES
        
        indices = []
        
        for feature in subset:
            index = features.index(feature)
            indices.append(index)
            
        indices.sort()
        
        names = []
        
        feature_indices = [-1] * len(features)
        
        for (i,index) in enumerate(indices):
            name = features[index]
            names.append(name)
            feature_indices[index] = i
            
        self.indices = indices
        self.names = names
        self.feature_indices = feature_indices
        
    def apply(self, behavior):
        
        result = []
        
        for index in self.indices:
            value = behavior[index]
            result.append(value)
            
        return result
         
filter = SubsetFilter(subset)
            
class NoveltySearch(object):

    def snap_shot(self, generation):
        pkl_path = self.pkl_path
        rarity_table_params = self.rarity_table_params
        most_rare = self.rarest.best
        
        if not os.path.isfile(pkl_path):
            pkl_file = open(pkl_path, "wb")
            pickle.dump(rarity_table_params, pkl_file)
            pickle.dump(filter.indices, pkl_file)
            pkl_file.close()
        
        pkl_file = open(pkl_path, "ab")
        pickle.dump(generation, pkl_file)
        pickle.dump(most_rare, pkl_file)
        pkl_file.close()
        

    def __init__(self, genome, params, evaluate, rarity_table_params, threshhold_control, seed=1):

        self.threshhold_control = threshhold_control

        self.rarity_table_params = rarity_table_params
        (sample_count, population_size, generations, bin_count) = rarity_table_params
        rarity_table = rarity_recognition.get_rarity_table(
            sample_count, population_size, generations, bin_count)

        self.rarity_table = rarity_table

        from datetime import datetime

        now = datetime.now()

        dir_name = "data"

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        feature_count = len(filter.names)
            
        file_name = "novelty_f%d_g%d_t%.1f_%04d.%02d.%02d_%02d:%02d:%02d" % (
            feature_count, generations, threshhold_control,
            now.year, now.month, now.day, now.hour, now.minute, now.second)

        self.csv_path = os.path.join(dir_name, file_name + ".csv")
        self.pkl_path = os.path.join(dir_name, file_name + ".pkl")

        # self.rarity_table.debug_print_table()
        # archive of past behaviors
        self.archive = []

        # archive of the *genomes* that represent those past behaviors
        self.garchive = []

        # evaluation function
        self.evaluate = evaluate

        # initialize the population
        self.population = NEAT.Population(genome, params, True, 1.0, seed)
        self.population.RNG.Seed(seed)

        self.rarest = NBest(100, nbest.filter_all_salients_match_bins)

        self.do_halt = False
        self.running = True

    def halt(self):
        self.do_halt = True

    def resume(self):
        self.do_halt = False

    def do_generations(self):

        evaluate = self.evaluate
        population = self.population

        # do the requested number of generations of evolution

        generation = -1

        while True:
        
            # entering next generation
        
            generation += 1
            
            if generation >= 1000:
                return
            
            # halt thread while asked to do so
            
            while self.do_halt:
                # let the outside know that we are not running
                self.running = False

            self.running = True
            
            
            genome_list = NEAT.GetGenomeList(population)

            # get the behavior for each genome in the population
            behavior_list = [filter.apply(evaluate(g)) for g in genome_list]
            
            def normalize(behavior):
                
                normalized = []
                
                for (i, feature) in enumerate(behavior):
                    index = filter.indices[i]
                    binner = self.rarity_table.binners[index]
                    mini = float(binner.minimum)
                    maxi = float(binner.maximum)
                    feat = float(feature)
                    norm = (feat - mini) / (maxi - mini)
                    normalized.append(norm) 
                    
                return normalized
                    
            # judge the novelty of a new indiviudal by all the behaviors of
            # current population + archive
            
            #compiled_array = numpy.array(self.archive + behavior_list)
            
            n_behavior_list = [normalize(b) for b in behavior_list]
            n_archive = [normalize(b) for b in self.archive]
            
            n_compiled = n_archive + n_behavior_list

            fitness_list = [calc_novelty(b, n_compiled)
                            for b in n_behavior_list]

            # randomly add one individual to archive per generation
            # you can do other things here... see original NS paper if
            # interested..
            idx = random.randint(0, len(behavior_list) - 1)
            self.archive.append(behavior_list[idx])
            self.garchive.append(NEAT.Genome(genome_list[idx]))

            # assign novelty as the fitness for each individual
            NEAT.ZipFitness(genome_list, fitness_list)

            print ">> rarity"
            # calculate rarity
            for (generation_id, behavior) in enumerate(behavior_list):
                print "  %d\r" % generation_id,
                sys.stdout.flush()
                
                genome = genome_list[generation_id]

                stats = make_rarity_stats(
                    genome,
                    behavior,
                    filter.indices,
                    generation,
                    generation_id,
                    self.rarity_table,
                    self.threshhold_control)
                
                for stat in stats:
                    
                    self.rarest.insert(stat)
            print ""
            print "<< rarity"
            
            print ("generation#%d" % (generation)),

            best_rarity = self.rarest.best_ranking()
            avg_rarity = self.rarest.avg_ranking()

            if best_rarity is not None:
                print "\tbest rarity: %f\taverage rarity: %f" % (best_rarity, avg_rarity)
            else:
                print ""

            # take a snapshot of the rarest

            if generation % 5 == 0:
                print "Taking a snapshot...",
                self.snap_shot(generation)
                print "...done!"


            # advance the population into the next generation
            
            population.Epoch()
            
def main(args):
    game = ball_keeper.init()

    # set up the parameters for MultiNEAT
    parameters = config.get_neat_parameters()

    # the initial genome which the population will be based on
    genome = config.get_neat_genome_seed(parameters)

    # the evaluation function
    def evaluate(genome):
        evaluation = ball_keeper.evaluate_genome(game, genome)
        behavior = evaluation.all_features()
        return behavior

    # a seed for some RNG
    seed = 10

    from flags import query_int, query_float

    sample_count = query_int("sample_count", 10000000)
    population_size = query_int("population_size", 10)
    generations = query_int("generations", 200)
    bin_count = query_int("bin_count", 20)
    threshhold_control = query_float("threshhold_control", 10.0)

    rarity_table_params = (sample_count, population_size, generations, bin_count)

    novelty_searcher = NoveltySearch(
        genome, parameters, evaluate, rarity_table_params, threshhold_control, seed)

    thread.start_new_thread(novelty_searcher.do_generations, ())

    while True:
        raw_input()

        novelty_searcher.halt()

        while novelty_searcher.running:
            pass

        do_exit = yesno.query("Do you want to exit?", "no")
        if do_exit:
            sys.exit(0)

        most_rare = novelty_searcher.rarest.best
        rarity_table = novelty_searcher.rarity_table

        print "[ RAREST BEHAVIORS ]"
        for (index, stats) in enumerate(most_rare):

            genome = stats.genome
            behavior = stats.behavior
            generation = stats.generation
            generation_id = stats.generation_id
            features = stats.features
            rarity = stats.rarity
            probability = stats.probability
            support = stats.support

            feature_list = list(features)
            total = rarity_table.data_size

            print "Individual #%d_%d" % (generation, generation_id)
            print "  behavior    : %s" % (str(behavior))
            print "  columns     : %s" % (str(features))
            print "  rarity      : %f" % (rarity)
            print "  probability : %f (%s of %d)" % (probability, str(support), total)
            print " "
            for feature in feature_list:
                
                idx = filter.feature_indices[feature]
                
            
                binr = rarity_table.binners[feature]
                desc = rarity_table.column_names[feature]
                val = behavior[idx]
                bin_index = binr.bin_value(val)
                (lower, upper) = binr.get_range(bin_index)
                print "    bin #%d of %s(%d) :: value: %s :: bin range: (%s to %s)" % (bin_index, desc, feature, str(val), str(lower), str(upper))
                print "-----------------------------"

            replay = yesno.query("Replay this individual?", "no")
            while replay:
                evaluation = ball_keeper.play_genome(game, genome)

                replay = yesno.query("Replay this individual?", "no")

            if index + 1 < len(most_rare):
                inspect_next = yesno.query("Inspect the next individual?")
                if not inspect_next:
                    break

        novelty_searcher.resume()

if __name__ == '__main__':
    main(sys.argv)
