from __future__ import print_function

import os 
import neat  
import visualize 
import pandas as pd 


iris_data = pd.read_csv("/Users/johnsaxon/test/github.com/learn-neat/meat/iris/testdata/iris_train.csv", header=None)

# print(iris_data.head(5))
# print(iris_data.columns)
# print(len(iris_data.columns))
# print(iris_data.index)

# print(iris_data[:,1])



# features = pd.DataFrame(iris_data, columns=[0,1,2,3])
# print(features.head(5))

# # print(features)

# labels = pd.DataFrame(iris_data, columns=[4])
# print(labels.head(5))

def traverse_iris():
    for i,row in iris_data.iterrows():
        if i>10:
            break
        # print(row)
        # print(row.values)
        # print(type(row.values))
        print(row.values[:4])
        print(row.values[4])



def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 4.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for i,row in iris_data.iterrows():
            output = net.activate(row.values[:4])
            # 这里需要改用cross entropy when task is classification
            # genome.fitness -= (output[0]-row[4])**2
            if output[0]!= row.values[4]:
                genome.fitness -= 1


def run(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation, 
                            config_file)

    # printObjectDict(config)

    printObjectDict(config.genome_config)

    # printObjectDict(neat.DefaultGenome)

    # printObjectDir(neat.DefaultGenome)

    setIO(config, 2, 1)

    printObjectDict(config.genome_config)
    

def setIO(config, num_inputs, num_outputs):
    assert isinstance(config, neat.Config) 

    config.genome_config.num_inputs = num_inputs
    config.genome_config.num_outputs = num_outputs

    config.genome_config.input_keys = [-i - 1 for i in range(config.genome_config.num_inputs)]
    config.genome_config.output_keys = [i for i in range(config.genome_config.num_outputs)]

def printObjectDict(obj):
    for k,v in obj.__dict__.items():
        print(k)
        print(v)
        print()


def printObjectDir(obj):
    print(dir(obj))




if __name__ == '__main__':
    # traverse_iris()

    # Detemin path to configuration file.This path manipulation is
    # here so that the script will successfully regardless of the 
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config_feedforward')
    run(config_path)
