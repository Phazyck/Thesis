"""
A tiny module, containing a custom data-structure
for keeping track of the N best objects.
"""

def filter_none(a,b):
    return False

_DEBUG = False
    
def filter_all_features_match(a,b):
    val_0 = tuple(a.get_value())
    val_1 = tuple(b.get_value())
    same_values = (val_0 == val_1)
    
    if _DEBUG and same_values:
        print "filter_all_features_match()"
        print "found match"
        print val_0
        print val_1
        raw_input()
        
    return same_values

def filter_all_salients_match(a,b):
    salient_0 = tuple(a.salient)
    salient_1 = tuple(b.salient)
    same_salients = (salient_0 == salient_1)
    
    val_0 = tuple(a.get_value())
    val_1 = tuple(b.get_value())
    same_values = (val_0 == val_1)
    
    if _DEBUG and same_salients and not same_values:
        print "filter_all_salients_match()"
        print "found match" 
        print val_0
        print val_1
        print salient_0
        print salient_1
        raw_input()
    
    return same_salients

def filter_all_salients_match_bins(a,b):
    val_0 = tuple(a.get_value())
    val_1 = tuple(b.get_value())
    same_values = (val_0 == val_1)
    
    if same_values:
        return True

    salient_0 = tuple(a.salient)
    salient_1 = tuple(b.salient)
    same_salients = (salient_0 == salient_1)
    
    if same_salients:
        return True
    
    bins_0 = tuple(a.salient_bin_indices)
    bins_1 = tuple(b.salient_bin_indices)
    same_bins = (bins_0 == bins_1)
    
    if _DEBUG and same_bins:
        print "filter_all_filter_all_salients_match_binssalients_match()"
        print "found match" 
        
        val_len = len(val_0)
        
        for i in range(len(val_0)):
            v0 = val_0[i]
            v1 = val_1[i]
            
            if v0 == v1:
                print "== %f" % (v0)
            else:
                print "!= %f | %f" % (v0, v1)
        
        print salient_0
        print salient_1
        print bins_0
        print bins_1
        
        print "same_values: %s" % (str(same_values))
        print "same_salients: %s" % (str(same_salients))
        print "same_bins: %s" % (str(same_bins))
        
        raw_input()
    
    return same_bins    

class NBest(object):
    """
    A custom data-structure
    for keeping track of the N best objects.
    """

    def __init__(self, n, filter=filter_none):
        self.best = []
        self.limit = n
        self.filter = filter
        
    def clear(self):
        self.best = []
        
    def insert(self, contender):
        """
        A function for inserting a new element into the collection.
        """
        filter = self.filter
        best = self.best
        limit = self.limit
        
        i = 0
        size = len(best)
        
        while i < size:
            previous = best[i]
            
            if filter(previous,contender):
                return
            
            rank_0 = previous.get_ranking()
            rank_1 = contender.get_ranking()
            
            if rank_1 > rank_0:
                best.insert(i, contender)
                break
                
            i += 1
            
        if(i == size):
            best.insert(i, contender)
            
        if(len(best) > limit):
            self.best = best[:limit]

    def best_ranking(self):
        """
        A function for getting the ranking
        of the highest-ranked object.
        """
        best = self.best
        
        if len(best) > 0:
            return best[0].get_ranking()
        else:
            return None
        

    def avg_ranking(self):
        best = self.best
        
        sum = 0.0
        count = 0.0
        
        for elm in best:
            sum += elm.get_ranking()
            count += 1.0
                
        return (sum / count)