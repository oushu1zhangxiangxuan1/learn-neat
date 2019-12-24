from __future__ import print_function

import os 
import neat  
import visualize 
import pandas as pd 
# import sklearn as 

from sklearn.preprocessing import  OneHotEncoder


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

def eval_genomes(genomes, config, encoder):
    for genome_id, genome in genomes:
        genome.fitness = 4.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for i,row in iris_data.iterrows():
            output = net.activate(row.values[:4])
            # 这里需要改用cross entropy when task is classification
            # genome.fitness -= (output[0]-row[4])**2

            print(output)

            encoder.transform(row.values[4])

            if output[0]!= row.values[4]:
                genome.fitness -= 1


def run(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation, 
                            config_file)

    # create the population, which is the top-level object fir a neat run
    p = neat.Population(config)

    #Add a stdout reporter to show progress in the terninal
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 30)

    #Display the winning genome.
    print('\n Best genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for i,row in iris_data.iterrows():
        output = winner_net.activate(row.values[:4])
        print('input {!r}, expected output {!r}, got {!r}'.format(
            row.values[:4], row.values[4], output
        ))

    # How does the node_names work?
    node_names = {-1:'A', -2:'B', -3:'C', -4:'D', 0:'label'}
    visualize.draw_net(config, winner, True, node_names=node_names)

    p = neat.Checkpointer.restore_checkpoint('neat-chepoint-4')
    p.run(eval_genomes, 10)


def test_run(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation, 
                            config_file)

    # create the population, which is the top-level object fir a neat run
    p = neat.Population(config)

    #Add a stdout reporter to show progress in the terninal
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 1)

    # #Display the winning genome.
    # print('\n Best genome:\n{!s}'.format(winner))

    # # Show output of the most fit genome against training data.
    # print('\nOutput:')
    # winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    # for i,row in iris_data.iterrows():
    #     output = winner_net.activate(row.values[:4])
    #     print('input {!r}, expected output {!r}, got {!r}'.format(
    #         row.values[:4], row.values[4], output
    #     ))

    # # How does the node_names work?
    # node_names = {-1:'A', -2:'B', -3:'C', -4:'D', 0:'label'}
    # visualize.draw_net(config, winner, True, node_names=node_names)

    # p = neat.Checkpointer.restore_checkpoint('neat-chepoint-4')
    # p.run(eval_genomes, 10)


def LabelEncoder(labels):
    oh_enc = OneHotEncoder(n_values=labels)




if __name__ == '__main__':
    # traverse_iris()

    # Detemin path to configuration file.This path manipulation is
    # here so that the script will successfully regardless of the 
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config_feedforward')
    # run(config_path)

    test_run(config_path)
