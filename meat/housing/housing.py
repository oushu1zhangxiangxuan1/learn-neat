from __future__ import print_function

import env
import os
import logging
import logging.handlers
# import traceback
# import math
import argparse

import neat
import pandas as pd
# import numpy as np


from meat.utils import visualize
from meat.utils.util import mkdir_p

env.touch

DEBUG = False

logger = logging.getLogger()

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter(
    '[%(funcName)s: %(lineno)s ] %(asctime)s %(levelname)s: %(message)s'))
stream_handler.setLevel(logging.DEBUG)
logger.addHandler(stream_handler)


def getHousingLabels(train_data):
    return train_data.iloc[-1].unique()


def traverse_housing():
    for i, row in train_data.iterrows():
        if i > 10:
            break
        print(row.values[:-1])
        print(row.values[-1])


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 4.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for i, row in train_data.iterrows():
            output = net.activate(row.values[:-1])
            # genome.fitness -= (output[0]-row[-1])**2
            # if TEST:
            #     print("output type: {}".format(type(output)))
            #     print("output: {}".format(output))
            loss_mse = (output[0]-row.values[-1])**2
            # if TEST:
            #     print("row.values[-1] type: {}".format(type(row.values[-1])))
            #     print("row.values[-1]: {}".format(row.values[-1]))
            #     print("loss_mse type: {}".format(type(loss_mse)))
            #     print("loss_mes: {}".format(loss_mse))
            genome.fitness -= loss_mse


def run(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # create the population, which is the top-level object fir a neat run
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terninal
    # p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(
        5, filename_prefix="checkpoint/{}/housing-ck".format(NAME)))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, GENS)

    # Display the winning genome.
    print('\n Best genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    predictNum = train_data.shape[0]
    train_loss = 0.0
    for i, row in train_data.iterrows():
        output = winner_net.activate(row.values[:-1])
        expect_out = row.values[-1]
        train_loss += (output-expect_out)**2
    train_avg_loss = train_loss/predictNum
    print("\nTrain avg loss: {}".format(train_avg_loss), flush=True)

    predictNum = test_data.shape[0]
    predict_loss = 0.0
    for i, row in test_data.iterrows():
        output = winner_net.activate(row.values[:-1])
        expect_out = row.values[-1]
        predict_loss += (output-expect_out)**2

    test_avg_loss = predict_loss/predictNum
    print("\nTest  avg loss: {}".format(test_avg_loss), flush=True)

    node_names = {
        -1: 'MedInc', -2: 'HouseAge', -3: 'AveRooms', -4: 'AveBedrms', -5: 'Population',
        -6: 'AveOccup', -7: 'Latitude', -8: 'Longitude',
        0: 'Price'
    }
    visualize.draw_net(config, winner, False,
                       node_names=node_names, filename="{}.gv".format(NAME))

    return train_avg_loss, test_avg_loss


def MultiMSE(src, target):
    assert isinstance(src, list)
    assert len(src) == len(target)
    assert isinstance(target, list)
    assert len(target) > 1

    return sum([(src[i]-target[i])**2 for i in range(len(src))])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.description = "Test neat on housing data."
    parser.add_argument("--train_data", help="train data file absolute path",
                        type=str, default='../../testdata/housing/housing_train.csv')
    parser.add_argument("--test_data", help="test data file absolute path",
                        type=str, default='../../testdata/housing/housing_test.csv')
    parser.add_argument(
        "--debug", help="train data file absolute path", type=bool, default=False)
    parser.add_argument(
        "--test", help="test data file absolute path", type=bool, default=False)
    parser.add_argument(
        "--name", help="file name to save checkpoint and graphviz", type=str, default="test")
    args = parser.parse_args()
    DEBUG = args.debug
    TEST = args.test
    NAME = args.name

    if DEBUG:
        print("\nargs:{}\n".format(args))

    train_data = pd.read_csv(args.train_data, header=None)
    test_data = pd.read_csv(args.test_data, header=None)

    # Detemin path to configuration file.This path manipulation is
    # here so that the script will successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    mkdir_p(os.path.join(local_dir, 'checkpoint', NAME))
    config_path = os.path.join(local_dir, 'config_feedforward')
    GENS = 100
    while True:
        if TEST:
            GENS = 1
        train_acc, test_acc = run(config_path)
        if TEST:
            break
        # if (train_acc > 0.975 and test_acc > 0.97) or GENS > 500:
        #     break
        GENS += 50
