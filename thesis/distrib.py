import sys

import sampling2

from sampling2 import Meta

import numpy as np

def plot3d(x, xlabel, y, ylabel, bin_count):

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import numpy as np

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
    from scipy.stats import gaussian_kde
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
     
def main(args):
    """
    The main entrypoint.
    """
    
    from flags import query_int

    sample_count = query_int("sample_count", 10000000)
    population_size = query_int("population_size", 10)
    generations = query_int("generations", 200)
    bin_count = query_int("bin_count", 10)
   
    sample_manager = sampling2.SampleManager(population_size, generations)

    column_names = sample_manager.meta.feature_names
    
    def get_feature_idx():
        for idx, name in enumerate(column_names):
            print "  %2d: %s" % (idx, name)
        
        idx = int(raw_input("> "))
        
        return idx, column_names[idx]
        
    print "Pick the 1st feature:"
    idx0, name0 = get_feature_idx()
    
    print "Pick the 2nd feature:"
    idx1, name1 = get_feature_idx()
    

    behavior_list = sample_manager.sample_reader(sample_count)
        
    array0 = np.zeros(sample_count)
    array1 = np.zeros(sample_count)
    
    for idx, behavior in enumerate(behavior_list):
        array0[idx] = behavior[idx0]
        array1[idx] = behavior[idx1]
    
    plot3d(array0, name0, array1, name1, bin_count)
        

if __name__ == "__main__":
    main(sys.argv)