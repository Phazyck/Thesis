import sys
from rarity_recognition import get_rarity_table, ValueBinner, RarityTable
from ball_keeper import FEATURE_NAMES
       
def RarityRecognizer(object):
    def __init__(self, means, std_devs):
        self.means = means
        self.std_devs = std_devs
    
    def get_rarity(feature, feature_indices):
        means = self.means
        std_devs = self.std_devs
        
        sqr_dist_sum = 0.0
        
        for (idx, feature) in enumerate(feature):
            feature_index = feature_indices[idx]
            
            mean = means[feature_index]
            std_dev = std_devs[feature_index]
            
            dist = (mean - feature) / std_dev
            
            sqr_dist_sum += (dist * dist)
            
        from math import sqrt
        
        rarity = sqrt(sqr_dist_sum)
        
        return rarity
                   
def make_recognizer(rarity_table):
    
    num_features = len(FEATURE_NAMES)
    
    means = []
    std_devs = []
    
    for i in range(num_features):
    
        name = FEATURE_NAMES[i]
        
        key = (i,)
        histogram = rarity_table.data[key]
        binner = rarity_table.binners[i]
        
        max_count = None
        mean = None
        
        values_and_counts = []
        
        
        for bin in histogram.keys():
            count = histogram[bin]
            
            (bin_idx,) = bin
            (lower,upper) = binner.get_range(bin_idx)
            value = float(lower + upper) * 0.5
            
            values_and_counts.append((value,count))
            
            if max_count is None or count > max_count:
                max_count = count
                mean = value
        
        means.append(mean)
        
        squared_diffs = 0.0
        total_counts = 0.0
        
        for (value,count) in values_and_counts:
            diff = mean - value
            squared_diff = diff * diff
            squared_diffs += squared_diff * count
            total_counts += count
            
        variance = squared_diffs / total_counts
        
        from math import sqrt
        
        std_dev = sqrt(variance)
        std_devs.append(std_dev)
        
    recognizer = RarityRecognizer(means, std_devs)
    return recognizer
            
def main(args):
    """
    The main entrypoint.
    """
    
    from flags import query_int

    sample_count = query_int("sample_count", 10000000)
    population_size = query_int("population_size", 10)
    generations = query_int("generations", 200)
    bin_count = query_int("bin_count", 20)
   
    print "Preparing rarity table..."

    table = get_rarity_table(
        sample_count,
        population_size,
        generations,
        bin_count)
        
    recognizer = make_recognizer(table)

if __name__ == "__main__":
    main(sys.argv)