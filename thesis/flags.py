import config

# utility class for parsing console arguments
class FlagParser(object):

    def __init__(
            self,
            sample_count=config.default_sample_count,
            population_size=config.default_population_size,
            generations=config.default_generations):
        self.sample_count = sample_count
        self.population_size = population_size
        self.generations = generations
        self.flags = {
            '--sample_count': self.parse_sample_count,
            '-sc': self.parse_sample_count,

            '--population_size': self.parse_population,
            '-ps': self.parse_population,

            '--generations': self.parse_generations,
            '-g': self.parse_generations,
        }

    def parse_sample_count(self, args):
        self.sample_count = int(args.pop())
        print "flag set: sample_count=%d" % self.sample_count

    def parse_population(self, args):
        self.population_size = int(args.pop())
        print "flag set: population_size=%d" % self.population_size

    def parse_generations(self, args):
        self.generations = int(args.pop())
        print "flag set: generations=%d" % self.generations

    def parse_flags(self, argv):
        args = list(reversed(argv))
        args.pop()

        while len(args) > 0:
            arg = args.pop()

            if arg in self.flags:
                handle_flag = self.flags[arg]
                handle_flag(args)
            else:
                print "warning, unknown flag: %s" % arg

class FlagQuerier(object):

    def __init__(
            self,
            sample_count=config.default_sample_count,
            population_size=config.default_population_size,
            generations=config.default_generations):
        self.sample_count = sample_count
        self.population_size = population_size
        self.generations = generations
        
        queries = [
            "sample_count",
            "population_size",
            "generations"]

        for query in queries:

            default = getattr(self, query)
            print ("set %s to (default = %s):" % (query, str(default))), 
            
            answer = raw_input()
            if len(answer) > 0:
                setattr(self, query, answer)

            value = getattr(self, query)
            print ('%s set to "%s"\n' % (query, str(value)))
            