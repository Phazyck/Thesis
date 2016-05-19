import os.path
import sys
import pickle
from achiever import RarityStats
from ball_keeper import FEATURE_NAMES
from rarity_recognition import get_rarity_table, RarityTable, ValueBinner

def get_file_path(file_name, dir_names=[]):
    
    path = ""
    
    for dir_name in dir_names:
        path = os.path.join(path, dir_name)
        if not os.path.exists(path):
            os.makedirs(path)
            
    path = os.path.join(path, file_name)
    
    return path
    
def get_full(pkl_path):
     pkl_file = open(pkl_path, "rb")
     
     def load():
        return pickle.load(pkl_file)
     
     rarity_table_params = load()
     (sample_count, population_size, generations, bin_count) = rarity_table_params
     filter_indices = load()
     rarity_table = get_rarity_table(sample_count, population_size, generations, bin_count)
     
     print "    ;    ;sample_count    ;%d" % sample_count
     print "    ;    ;population_size ;%d" % population_size
     print "    ;    ;generations     ;%d" % generations
     print "    ;    ;bin_count       ;%d" % bin_count
     print " "
     
     while True:
        try:
            generation = load()
            print "###;    ;generation;%d" % generation
            print " "
            most_rare = load()
            for r in most_rare:
                #print r.genome
                
                print "---;    ;subject;%d" % (r.generation_id)
                print "   ;    ;generation;%d" % (r.generation)
                print " "
                
                print "   ;    ;rarity;%f" % (r.rarity)
                print " "
                
                print "   ;    ;probability;%f;%%" % (100 * r.probability)
                print " "
                
                print "   ;    ;support;%s" % (str(r.support))
                print "   ;    ;of;%d" % (sample_count)
                print " "
                
                bs = [(False, feature) for feature in r.behavior]
                for feature in r.features:
                    idx = filter_indices.index(feature)
                    (_, value) = bs[idx]
                    bs[idx] = (True, value)
                    
                
                for (idx,(t,b)) in enumerate(bs):
                    i = filter_indices[idx]
                    binner = rarity_table.binners[i]
                    bin_index = binner.bin_value(b)
                    (lower,upper) = binner.get_range(bin_index)
                    
                    if t:
                        print "!!!;",
                    else:
                        print "   ;",
                    
                    feature_name = FEATURE_NAMES[i]
                    print "%3d;%s;%.5f;  ;bin[%d];%f;to;%f" % (i,feature_name, b, bin_index, lower, upper)
                    
                print " "
        except EOFError:
            break
     
     pkl_file.close()
     
def get_latest(pkl_path):
    pkl_file = open(pkl_path, "rb")
     
    def load():
        return pickle.load(pkl_file)
     
    rarity_table_params = load()
    (sample_count, population_size, generations, bin_count) = rarity_table_params
    filter_indices = load()
    rarity_table = get_rarity_table(sample_count, population_size, generations, bin_count)

    print "    ;    ;sample_count    ;%d" % sample_count
    print "    ;    ;population_size ;%d" % population_size
    print "    ;    ;generations     ;%d" % generations
    print "    ;    ;bin_count       ;%d" % bin_count
    print " "

    latest_generation = None
    latest_most_rare = None

    while True:
        try:
            generation = load()
            most_rare = load()
            
            if generation != None and most_rare != None:
                latest_generation = generation
                latest_most_rare = most_rare
        except EOFError:
            break
     
    print "###;    ;generation;%d" % generation
    print " "
    
    for r in most_rare:
        #print r.genome
        
        print "---;    ;subject;%d" % (r.generation_id)
        print "   ;    ;generation;%d" % (r.generation)
        print " "
        
        print "   ;    ;rarity;%f" % (r.rarity)
        print " "
        
        print "   ;    ;probability;%f;%%" % (100 * r.probability)
        print " "
        
        print "   ;    ;support;%s" % (str(r.support))
        print "   ;    ;of;%d" % (sample_count)
        print " "
        
        bs = [(False, feature) for feature in r.behavior]
        for feature in r.features:
            idx = filter_indices.index(feature)
            (_, value) = bs[idx]
            bs[idx] = (True, value)
            
        
        for (idx,(t,b)) in enumerate(bs):
            i = filter_indices[idx]
            binner = rarity_table.binners[i]
            bin_index = binner.bin_value(b)
            (lower,upper) = binner.get_range(bin_index)
            
            if t:
                print "!!!;",
            else:
                print "   ;",
            
            feature_name = FEATURE_NAMES[i]
            print "%3d;%s;%.5f;  ;bin[%d];%f;to;%f" % (i,feature_name, b, bin_index, lower, upper)
     
    print " "
     
    pkl_file.close()
     
def get_progression(pkl_path):
     pkl_file = open(pkl_path, "rb")
     
     def load():
        return pickle.load(pkl_file)
     
     rarity_table_params = load()
     (sample_count, population_size, generations, bin_count) = rarity_table_params
     filter_indices = load()
     rarity_table = get_rarity_table(sample_count, population_size, generations, bin_count)
     
     print "    ;    ;sample_count    ;%d" % sample_count
     print "    ;    ;population_size ;%d" % population_size
     print "    ;    ;generations     ;%d" % generations
     print "    ;    ;bin_count       ;%d" % bin_count
     print " "
     print "generation;max rarity;average rarity"
     
     while True:
        try:
            generation = load()
            most_rare = load()
            max_rarity = None
            sum_rarity = 0.0
            count = 0.0
            for r in most_rare:
                rarity = r.rarity
                
                if max_rarity is None or rarity > max_rarity:
                    max_rarity = rarity
                    
                sum_rarity += rarity
                count += 1.0
                
            average_rarity = sum_rarity / count
            print "%d;%f;%f" % (generation, max_rarity, average_rarity)
            
        except EOFError:
            break
     
     pkl_file.close()
     
def get_salients(pkl_path):
    pkl_file = open(pkl_path, "rb")
     
    def load():
        return pickle.load(pkl_file)
     
    rarity_table_params = load()
    (sample_count, population_size, generations, bin_count) = rarity_table_params
    filter_indices = load()
    rarity_table = get_rarity_table(sample_count, population_size, generations, bin_count)
     
    print "    ;    ;sample_count    ;%d" % sample_count
    print "    ;    ;population_size ;%d" % population_size
    print "    ;    ;generations     ;%d" % generations
    print "    ;    ;bin_count       ;%d" % bin_count
    print " "
     
    salients_total = {}
    salients_generations = []
     
    while True:
        try:
            salients_generation = {}
            
            generation = load()
            most_rare = load()
            
            for r in most_rare:
                
                salient = len(r.salient)
                def increment(key, table):
                    if key in table:
                        table[key] += 1
                    else:
                        table[key] = 1
                
                increment(salient, salients_generation)
                increment(salient, salients_total)
            
            salients_generations.append((generation, salients_generation))
        except EOFError:
            break
    
    keys = salients_total.keys()
    key_range = range(min(keys), max(keys)+1)
            
    print "generation",
    
    for i in key_range:
        print (";%d-sets" % i),
    
    print ""     
    
    for (generation, salients) in salients_generations:
        print generation,
        
        for i in key_range:
            value = 0
            if i in salients:
                value = salients[i]
                
            print (";%d" % value),
            
        print ""
           
    pkl_file.close()     
     
def get_summary(pkl_path):
    pkl_file = open(pkl_path, "rb")
     
    def load():
        return pickle.load(pkl_file)
     
    rarity_table_params = load()
    (sample_count, population_size, generations, bin_count) = rarity_table_params
    filter_indices = load()
    rarity_table = get_rarity_table(sample_count, population_size, generations, bin_count)
    
    latest_generation = None
    latest_most_rare = None
    
    while True:
        try:
            generation = load()
            most_rare = load()
            
            if generation != None and most_rare != None:
                latest_generation = generation
                latest_most_rare = most_rare
        except EOFError:
            break
    
    print "    ;    ;sample_count    ;%d" % sample_count
    print "    ;    ;population_size ;%d" % population_size
    print "    ;    ;generations     ;%d" % generations
    print "    ;    ;bin_count       ;%d" % bin_count
    print " "
            
    print "subject;generation;support;rarity",
    for i in range(3):
        n = i + 1
        print (";feature %d;value %d") % (n,n),
            
    print " "
    
    for stats in latest_most_rare:
        subject = stats.generation_id
        generation = stats.generation
        support = stats.support
        probability = stats.probability
        rarity = stats.rarity
        
        salients = stats.salient
        salient_bin_indices = stats.salient_bin_indices
        
        num_salients = len(salients)
        
        print ("%d;%d;%s;%f" % (subject, generation, support, rarity)),
        
        for idx in range(num_salients):
            salient = salients[idx]
            (feature_idx,_) = salient_bin_indices[idx]
            feature_name = FEATURE_NAMES[feature_idx]
            print (";%s" % (str(feature_name))),
            print (";%s" % (str(salient))),
        
        
        print " "
           
    pkl_file.close()      
     
def _main(argv):
    argc = len(argv)
    
    if argc != 3:
        print "usage: python data2csv.py [full|progression|latest|salients] <pkl_path>"
        exit(0)
    
    mode = argv[1]
    pkl_path = argv[2]
    
    if mode == "full":
        get_full(pkl_path)
    elif mode == "progression":
        get_progression(pkl_path)
    elif mode == "latest":
        get_latest(pkl_path)
    elif mode == "salients":
        get_salients(pkl_path)
    elif mode == "summary":
        get_summary(pkl_path)
    else:
        print "usage: python data2csv.py [full|progression] <pkl_path>"
    
if __name__ == '__main__':
    _main(sys.argv)