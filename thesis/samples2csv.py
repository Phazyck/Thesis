from sampling2 import SampleManager

import csv

def _main():
    population_size = 64
    generations = 32
    sample_count = 1000
    
    manager = SampleManager(population_size, generations)
    reader = manager.sample_reader(sample_count)
    
    names = manager.feature_names()
    
    filename = "samples.csv"
    
    with open(filename, "wb") as csv_file:
        csv_writer = csv.writer(
            csv_file, 
            delimiter=';', 
            quotechar='"', 
            quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(names)
        for sample in reader:
            csv_writer.writerow(sample)
        
if __name__ == '__main__':
    _main()