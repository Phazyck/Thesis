import sys
import os.path
import string

import sampling2

from sampling2 import Meta

import time

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde

import dill
import pickle
import yesno
from math import log

class KdeRecognizer(object):
    
    def __init__(self, 
            samples, 
            sample_count,
            population_size,
            generations):
        self.feature_count = len(samples)
        
        self.base_samples = samples
        self.base_sample_count = sample_count
        
        self.additinal_samples = [[]] * self.feature_count
        self.additinal_sample_count = 0
        
        self.population_size = population_size
        self.generations = generations
        self.additional_samples = [[]] * self.feature_count
        self.invalidate_kdes()
        
    def get_probability(self, value, feature_id):
        
        kdes = self.kdes
        kde = kdes[feature_id]
        
        if kde is None:
            base_data = self.base_samples[feature_id]
            additional_data = self.additional_samples[feature_id]
            
            data = base_data + additional_data
            kde = gaussian_kde(data)
            kde.covariance_factor = lambda : 1
            kde._compute_covariance()
            kdes[feature_id] = kde
            
        probability = kde(value)
        
        return probability
        
    def invalidate_kdes(self):
        self.kdes = [None] * self.feature_count
        
    def total_sample_count(self):
        c1 = self.base_sample_count
        c2 = self.additional_sample_count
        return c1 + c2
        
    def set_samples(self, behavior_list):
    
        additional_samples = [[]] * self.feature_count
        self.additinal_sample_count = len(behavior_list)
        
        for behavior in behavior_list:
            for list_idx, value in enumerate(behavior):
                additional_samples[list_idx].append(value)
        
        self.invalidate_kdes()
    
def get_kdes_path(sample_count, population_size, generations):
    dir_name = "kde_cache"

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        
    formatter = string.Formatter()
        
    file_name = formatter.format(
        "cache_{0}_{1}_{2}.pkl",
        sample_count,
        population_size,
        generations)
        
    file_path = os.path.join(dir_name, file_name)

    return file_path
    
def load_kdes(file_path):
    
    try:
        pkl_file = open(file_path, "rb")
        
        kdes = pickle.load(pkl_file)
        
        pkl_file.close()

        return kdes
    except IOError:
        return None
    
def save_kdes(kdes, file_path):

    pkl_file = open(file_path, "wb")

    pickle.dump(kdes, pkl_file)

    pkl_file.close()
    
def get_kdes(
        sample_count,
        population_size,
        generations):
        
    kdes_path = get_kdes_path(sample_count, population_size, generations)

    print "Loading kdes..."

    kdes = load_kdes(kdes_path)

    if kdes is None:
        print "...could not find cached kdes."
        print "Creating new kdes..."
        kdes = build_kdes(sample_count, population_size, generations, kdes_path)
        print "...caching kdes..."
        save_kdes(kdes, kdes_path)

    return kdes
    
def build_kdes(sample_count, population_size, generations, kdes_path):

    sample_manager = sampling2.SampleManager(population_size, generations)

    column_names = sample_manager.meta.feature_names
    
    behavior_list = sample_manager.sample_reader(sample_count)
        
    list_of_lists = [[]] * len(column_names)
    
    for idx, behavior in enumerate(behavior_list):
        print "\r%d" % idx,
        sys.stdout.flush()
        
        for list_idx, value in enumerate(behavior):
            list_of_lists[list_idx].append(value)
    
    kdes = KdeRecognizer(list_of_lists, 
            sample_count,
            population_size,
            generations)
    
    return kdes
    
def inspect(x, xlabel, bin_count):

    x_len = len(x)
    
    kde_time(x, x_len/100)
    kde_time(x, x_len)

    fig = plt.figure()
    data = x
    density = gaussian_kde(data)
    
    dmin = data.min()
    dmax = data.max()
    
    dwid = dmax - dmin
    
    xs = np.arange(dmin - dwid*0.5, dmax + dwid*0.5, dwid * 0.01)
    density.covariance_factor = lambda : 1
    density._compute_covariance()
    
    # histogram for x
    ax = fig.add_subplot(211, xlabel=xlabel, ylabel="count")
    
    ax.set_yscale('log', basey=10)
    ax.plot(xs, density(xs))
    
    ones = np.ones(len(xs))
    ax.plot(xs, ones, color='red')
    ax.hist(x, bin_count, log=True, color='green', facecolor='green', alpha=0.8)
    # pdf for x
    ax = fig.add_subplot(212, xlabel=xlabel, ylabel="count")
    ax.plot(xs, density(xs))
    ax.plot(xs, ones, color='red')
    
    ax.hist(data, normed=True, bins=bin_count)
    
def plot3d(x, xlabel, y, ylabel, bin_count):

    
    fig = plt.figure()
    ax = fig.add_subplot(231, projection='3d', xlabel=xlabel, ylabel=ylabel)
    
    hist, xedges, yedges = np.histogram2d(x, y, bins=bin_count)
    hist = np.log10(hist)

    # elements = (len(xedges) - 1) * (len(yedges) - 1)
    # xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25)

    # xpos = xpos.flatten()
    # ypos = ypos.flatten()
    # zpos = np.zeros(elements)
    # dx = 0.5 * np.ones_like(zpos)
    # dy = dx.copy()
    # dz = hist.flatten()
    

    # ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')
    
    # xpos, ypos = np.meshgrid(xedges[:-1]+xedges[1:], yedges[:-1]+yedges[1:])

    # xpos = xpos.flatten()/2.
    # ypos = ypos.flatten()/2.
    # zpos = np.zeros_like (xpos)

    # dx = xedges [1] - xedges [0]
    # dy = yedges [1] - yedges [0]
    # dz = hist.flatten()

    # ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')
    
    # The start of each bucket.
    xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1])

    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros_like(xpos)

    # The width of each bucket.
    dx, dy = np.meshgrid(xedges[1:] - xedges[:-1], yedges[1:] - yedges[:-1])

    dx = dx.flatten()
    dy = dy.flatten()
    dz = hist.flatten()

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')
    
    # histogram for x
    ax = fig.add_subplot(232, xlabel=xlabel, ylabel="count")
    
    ax.set_yscale('log', basey=10)
    ax.hist(x, bin_count, log=True, color='green', facecolor='green', alpha=0.8)
    
    # histogram for y
    ax = fig.add_subplot(233, xlabel=ylabel, ylabel="count")
    
    ax.set_yscale('log', basey=10)
    ax.hist(y, bin_count, log=True, color='green', facecolor='green', alpha=0.8)
    
    # -------------------
    # m1 = x
    # m2 = y

    # from scipy import stats
    
    # xmin = m1.min()
    # xmax = m1.max()
    # ymin = m2.min()
    # ymax = m2.max()

    # # Perform a kernel density estimate on the data
    
    # X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    # positions = np.vstack([X.ravel(), Y.ravel()])
    # values = np.vstack([m1, m2])
    # kernel = stats.gaussian_kde(values)
    # Z = np.reshape(kernel(positions).T, X.shape)
    
    # cmin = min(xmin, ymin)
    # cmax = max(xmax, ymax)

    # # Plot the results
    # ax = fig.add_subplot(224)
    # ax.autoscale_view()
    # ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r, extent=[cmin, cmax, cmin, cmax])
    # ax.plot(m1, m2, 'k.', markersize=2)
    # ax.set_xlim([cmin, cmax])
    # ax.set_ylim([cmin, cmax])
    
    
    # -----------------------------
    
    data = x
    density = gaussian_kde(data)
    
    ax = fig.add_subplot(235)
    
    xs = np.arange(data.min(), data.max(), .1)
    density.covariance_factor = lambda : 0.25
    density._compute_covariance()
    ax.plot(xs, density(xs))
    ax.hist(data, normed=True, bins=bin_count)
    
    data = y
    density = gaussian_kde(data)
    
    ax = fig.add_subplot(236)
    
    xs = np.arange(data.min(), data.max(), .1)
    density.covariance_factor = lambda : 0.25
    density._compute_covariance()
    ax.plot(xs, density(xs))
    ax.hist(data, normed=True, bins=bin_count)
     
    plt.show()
     
def test(args):
    """
    The main entrypoint.
    """
    from flags import query_int

    sample_count = query_int("sample_count", 10000000)
    population_size = query_int("population_size", 10)
    generations = query_int("generations", 200)
    bin_count = query_int("bin_count", 40)
   
    sample_manager = sampling2.SampleManager(population_size, generations)

    column_names = sample_manager.meta.feature_names
    
    for idx, name in enumerate(column_names):
        print idx, name
    
    behavior_list = sample_manager.sample_reader(sample_count)
        
    arrays = []
    for _ in column_names:
        array = np.zeros(sample_count)
        arrays.append(array)
    
    print "0",
    for idx, behavior in enumerate(behavior_list):
        print "\r%d" % idx,
        sys.stdout.flush()
        
        for arr_idx, value in enumerate(behavior):
            array = arrays[arr_idx]
            array[idx] = value
        
    print ""
    
    for idx in range(1):
        print idx
        array = arrays[idx]
        name = column_names[idx]
        inspect(array, name, bin_count)
        plt.show()
    
    print "done"
    
    
    raw_input()
    
    exit()
    
    def get_feature_idx():
        for idx, name in enumerate(column_names):
            print "  %2d: %s" % (idx, name)
        
        idx = int(raw_input("> "))
        
        return idx, column_names[idx]
        
    print "Pick the 1st feature:"
    idx0, name0 = get_feature_idx()
    
    print "Pick the 2nd feature:"
    idx1, name1 = get_feature_idx()
    
    array0 = arrays[idx0]
    array1 = arrays[idx1]
    
    
    
    plot3d(array0, name0, array1, name1, bin_count)
        

def main(args):
    """
    The main entrypoint.
    """
    
    from flags import query_int

    sample_count = query_int("sample_count", 1000000)
    population_size = query_int("population_size", 10)
    generations = query_int("generations", 200)
   
    print "Preparing kdes..."

    kdes = get_kdes(
        sample_count,
        population_size,
        generations)

    print "...kdes ready!"

    inspect_kdes = yesno.query("Do you wish to inspect the kdes?")

    if inspect_kdes:
        print "Sorry, this is not supported at the moment..."

if __name__ == "__main__":
    main(sys.argv)
