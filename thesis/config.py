import MultiNEAT as NEAT

ball_start_x = 100
ball_start_velocity_x = 50
player_start_x = 200

default_sample_count = 10000000
default_population_size = 64
default_generations = 32

def get_neat_parameters():
    params = NEAT.Parameters()
    
    ## Basic parameters
    
    params.PopulationSize = 150
    params.MinSpecies = 5
    
    ## GA Parameters
    
    params.YoungAgeTreshold = 15
    params.SpeciesMaxStagnation = 100
    params.OldAgeTreshold = 35
    params.CrossoverRate = 0.5
    params.OverallMutationRate = 0.02
    params.RouletteWheelSelection = True
    
    ## Structural Mutation parameters
    params.MutateAddLinkProb = 0.02
    params.RecurrentProb = 0.0
    
    ## Parameter Mutation parameters
    
    params.MutateWeightsSevereProb = 0.5
    params.WeightMutationRate = 0.75
    params.WeightReplacementMaxPower = 5.0
    params.MaxWeight = 20
    
    ## Speciation parameters
    
    params.DivisionThreshold = 0.5
    params.VarianceThreshold = 0.03
    params.MaxDepth = 4
    params.CPPN_Bias = -3.0
    params.Width = 1.
    params.Height = 1.
    params.Leo = True
    params.LeoThreshold = 0.3
    params.LeoSeed = True
    params.GeometrySeed = True
    params.Elitism = 0.1
    
    params.MutateWeightsSevereProb = 0.01
    
    return params
    
def get_neat_genome_seed(parameters):

    genome_id = 0
    num_inputs = 6
    num_hidden = 0  # ignored for seed_type == 0
    # specifies number of hidden units if seed_type == 1
    num_outputs = 2
    fs_neat = False
    output_act_type = NEAT.ActivationFunction.TANH
    hidden_act_type = NEAT.ActivationFunction.UNSIGNED_SIGMOID
    seed_type = 0
    genome = NEAT.Genome(genome_id, num_inputs, num_hidden, num_outputs,
                       fs_neat, output_act_type, hidden_act_type,
                       seed_type, parameters)

    return genome    
