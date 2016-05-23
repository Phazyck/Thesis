import sys
from rarity_recognition import get_rarity_table, ValueBinner, RarityTable
from ball_keeper import FEATURE_NAMES
       
rr2_repr_fmt = """
{
    behavior = %s,
    sailent = %s,
    salient_feature_index = %s,
    rarity = %s,
    feature_name = %s
}
"""
       
class RarityStats2(object):

    def __init__(self, 
        behavior, 
        salient, 
        salient_feature_index, 
        rarity, 
        feature_name):
        
        self.behavior = behavior
        self.salient = salient
        self.salient_feature_index = salient_feature_index
        self.rarity = rarity
        self.feature_name = feature_name
        
    def __str__(self):
        
        return (rr2_repr_fmt % (self.behavior,
            self.salient,
            self.salient_feature_index,
            self.rarity,
            self.feature_name))
       
class RarityRecognizer2(object):

    def __init__(self, means, std_devs):
        self.means = means
        self.std_devs = std_devs
    
    def get_rarity(self, feature, feature_indices):
        # print "\n>>> get_rarity\n"
    
        means = self.means
        std_devs = self.std_devs
        
        sqr_dist_sum = 0.0
        
        for (idx, feature) in enumerate(feature):
        
            # print "feature = %f" % feature
            
            feature_index = feature_indices[idx]
            # print "feature_index = %d" % feature_index
            # print "feature_name = %s" % FEATURE_NAMES[feature_index]
            
            mean = means[feature_index]
            
            # print "mean = %f" % mean
            std_dev = std_devs[feature_index]
            # print "std_dev = %f" % std_dev
            
            dist = (mean - feature) / std_dev
            
            sqr_dist_sum += (dist * dist)
            
        from math import sqrt
        
        rarity = sqrt(sqr_dist_sum)
        
        # print "rarity = %f" % rarity
        
        # print "\n<<< get_rarity\n"
            
        return rarity
        
    def get_rarity_stats(self, behavior, feature_indices):
        
        # print "-" * 64
        # print "\n>>> get_rarity_stats\n"
        
        # print "behavior = %s" % str(behavior)
        
        num_features = len(behavior)
        
        best_rarity = None
        best_values = []
        best_feature_indices = []
        
        for idx in range(num_features):
            # print "idx = %d" % idx 
            value = (behavior[idx],)
            # print "value = %s" % str(value)
            feature_idx = (feature_indices[idx],)
            # print "feature_idx = %s" % str(feature_idx)
            
            rarity = self.get_rarity(value, feature_idx)
            
            # print "rarity = %s" % str(rarity)
            
            if best_rarity is None or rarity > best_rarity:
                best_rarity = rarity
                # print "new best: %f" % best_rarity
                best_values = [value]
                best_feature_indices = [feature_idx]
            elif rarity == best_rarity:
                # print "expanding best"
                best_values.append(value)
                best_feature_indices.append(feature_idx)
                
        rarity_stats_list = []
        
        num_stats = len(best_values)
        
        # print "num_stats = %d" % num_stats
        
        for idx in range(num_stats):
            # print "idx = %d" % idx
            best_value = best_values[idx]
            
            best_feature_index = best_feature_indices[idx]
            
            best_feature_name = tuple(FEATURE_NAMES[idx] for idx in best_feature_index)
            
            # print "best_value = %s" % str(best_value)
            # print "best_feature_index = %s" % str(best_feature_index)
            # print "best_feature_name = %s" % str(best_feature_name)
                
            rarity_stats = RarityStats2(
                behavior, best_value, best_feature_index, best_rarity, best_feature_name)
                
            rarity_stats_list.append(rarity_stats)
            
        # print "\n"
        # print "best rarity = %f" % best_rarity
        
        # print "results (%d)" % len(rarity_stats_list)
        
        # for stat in rarity_stats_list:
        #     print "%s" % stat
            
        # print "\n<<< get_rarity_stat\n"
        # print "-" * 64
        
        # raw_input()
            
        return rarity_stats_list
                 
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
        
    recognizer = RarityRecognizer2(means, std_devs)
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