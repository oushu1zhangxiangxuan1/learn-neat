from __future__ import print_function

import env
import os 
import logging
import logging.handlers
import traceback
import math
import argparse

import neat  
import pandas as pd 
import numpy as np
from sklearn.preprocessing import OneHotEncoder


from meat.utils import visualize 
from meat.utils.util import mkdir_p

DEBUG = False

logger = logging.getLogger()

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter(
    '[%(funcName)s: %(lineno)s ] %(asctime)s %(levelname)s: %(message)s'))
stream_handler.setLevel(logging.DEBUG)
logger.addHandler(stream_handler)

def getIrisLabels(train_data):
    return train_data[5].unique()


def traverse_iris():
    for i,row in train_data.iterrows():
        if i>10:
            break
        print(row.values[:-1])
        print(row.values[-1])


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 4.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for i,row in train_data.iterrows():
            output = net.activate(row.values[:-1])
            # 这里需要改用cross entropy when task is classification
            # genome.fitness -= (output[0]-row[-1])**2
            if DEBUG:
                print(output)
                print(row.values[-1])

            label_encoded = zxxEncoder.transform(np.reshape(row.values[-1], (-1,1))).toarray()
            # res = np.reshape(res, (-1))
            label_encoded = np.ravel(label_encoded).tolist()
            if DEBUG:
                print("label: {}".format(row.values[-1]))
                print("after encoding: ")
                print(label_encoded)

            soft_res = ZxxSoftmax(output)
            if DEBUG:
                print("soft_res: {}".format(soft_res))

            loss_cross = ZxxCrossEntropy(soft_res,label_encoded)
            loss_mse = MultiMSE(soft_res, label_encoded)

            # if output[0]!= row.values[-1]:
            genome.fitness -= loss_cross


def run(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation, 
                            config_file)

    # create the population, which is the top-level object fir a neat run
    p = neat.Population(config)

    #Add a stdout reporter to show progress in the terninal
    # p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5, filename_prefix="checkpoint/{}/iris-ck".format(NAME)))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, GENS)

    #Display the winning genome.
    print('\n Best genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    predictNum = train_data.shape[0]
    predictRight = 0.0
    for i,row in train_data.iterrows():
        output = winner_net.activate(row.values[:-1])
        if TEST:
            print("output:\n")
            print(output)
            print("output reshape:\n")
            print(np.reshape(output, (-1,1)))
        output_str = zxxEncoder.inverse_transform(np.reshape(output, (-1,3)))

        if TEST:
            print("output_str:\n")
            print(output_str)

        expect_out = row.values[-1]

        if output_str == expect_out:
            predictRight += 1

        # print('expected index {!r}, got {!r}     expected output {!r}, got {!r}   '.format(
        #     index, expect_index, expect_out, output
        # ))
    train_accuracy = predictRight/predictNum
    print("\nTrain accuracy: {}".format(train_accuracy), flush=True)

    predictNum = test_data.shape[0]
    predictRight = 0.0
    for i,row in test_data.iterrows():
        output = winner_net.activate(row.values[:-1])
        output_str = zxxEncoder.inverse_transform(np.reshape(output, (-1,3)))
        expect_out = row.values[-1]

        if output_str == expect_out:
            predictRight += 1

    test_accuracy = predictRight/predictNum
    print("\nTest  accuracy: {}".format(test_accuracy), flush=True)

    # How does the node_names work?
    node_names = {-1:'Futile Input', -2:'B', -3:'C', -4:'D', -5:'E', 0:'label_0', 1:'label_1', 2:'label_2'}
    visualize.draw_net(config, winner, False, node_names=node_names, filename="{}.gv".format(NAME))

    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    # p.run(eval_genomes, 1)

    return train_accuracy, test_accuracy


def ZxxSoftmax(src):
    assert isinstance(src, list)
    assert len(src)>1
    src_max = max(src)

    # TODO: add a sigmoid to aviod OverflowError
    denominator = sum([math.e ** (i-src_max) for i in src])
    return [(math.e ** (i-src_max))/denominator for i in src]

def ZxxCrossEntropy(src, target):
    if DEBUG:
        print("src: {}".format(src))
        print("target: {}".format(target))
        print("src sum: {}".format(sum(src)))
    assert isinstance(src, list)
    assert len(src)==len(target)
    assert isinstance(target, list)
    assert len(target)>1
    # assert int(sum(src))==1.0
    # assert int(sum(target)) == 1.0
    # assert target.index(1)>0
    try:
        i = target.index(1)
        loss = -math.log(src[i]+10**-10)
        return loss
    except Exception as e:
        logging.error("src is {}".format(src))
        logging.error("target is {}".format(target))
        i = target.index(1)
        logging.error("i is {}".format(i))
        logging.error("src[i] is {}".format(src[i]))
        traceback.format_exc()
        raise e

def MultiMSE(src, target):
    assert isinstance(src, list)
    assert len(src)==len(target)
    assert isinstance(target, list)
    assert len(target)>1
    # assert sum(src)==1
    # assert sum(target) == 1

    return sum([(src[i]-target[i])**2 for i in range(len(src))])





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.description = "Test neat on iris data."
    parser.add_argument("--train_data", help="train data file absolute path", type=str, default='testdata/iris_train.csv')
    parser.add_argument("--test_data", help="test data file absolute path", type=str, default='testdata/iris_test.csv')
    parser.add_argument("--debug", help="train data file absolute path", type=bool, default=False)
    parser.add_argument("--test", help="test data file absolute path", type=bool, default=False)
    parser.add_argument("--name", help="file name to save checkpoint and graphviz", type=str, default="test")
    args = parser.parse_args()
    DEBUG = args.debug
    TEST  = args.test
    NAME = args.name

    if DEBUG:
        print("\nargs:{}\n".format(args))

    train_data = pd.read_csv(args.train_data, header=None)
    test_data = pd.read_csv(args.test_data, header=None)

    labels = getIrisLabels(train_data)

    # print("Iris labels: {}".format(labels))

    zxxEncoder = OneHotEncoder().fit(np.reshape(labels, (-1,1)))

    # TODO:
    # zxxEncoder = OneHotEncoder().fit(train_data[-1])

    # traverse_iris()

    # Detemin path to configuration file.This path manipulation is
    # here so that the script will successfully regardless of the 
    # current working directory.
    local_dir = os.path.dirname(__file__)
    mkdir_p(os.path.join(local_dir,'checkpoint', NAME))
    config_path = os.path.join(local_dir, 'config_feedforward')
    GENS=100
    while True:
        if TEST:
            GENS=1
        train_acc, test_acc = run(config_path)
        if TEST:
            break
        if (train_acc > 0.975 and test_acc > 0.97)  or GENS > 500:
            break
        GENS += 50
