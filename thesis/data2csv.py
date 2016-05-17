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
    
def load_snap_shot(pkl_path):
     pkl_file = open(pkl_path, "rb")
     
     def load():
        return pickle.load(pkl_file)
     
     rarity_table_params = load()
     (sample_count, population_size, generations) = rarity_table_params
     filter_indices = load()
     rarity_table = get_rarity_table(sample_count, population_size, generations)
     
     print "    ;    ;sample_count    ;%d" % sample_count
     print "    ;    ;population_size ;%d" % population_size
     print "    ;    ;generations     ;%d" % generations
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
     
def _main(argv):
    argc = len(argv)
    
    if argc != 2:
        print "usage: python data2csv.py <pkl_path>"
        exit(0)
    
    pkl_path = argv[1]
    load_snap_shot(pkl_path)
    
if __name__ == '__main__':
    _main(sys.argv)