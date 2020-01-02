from __future__ import print_function

import os 
import neat  
import visualize 
import pandas as pd 
import traceback
import math
# import sklearn as 

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder

import numpy as np

DEBUG = False

# from . import init


# from meat.iris.one_hot_test import ZxxEncoder

class ZxxEncoder():
    def __init__(self, labels):
        if DEBUG:
            print(labels)
        assert labels is not None
        self.le = LabelEncoder()
        self.le.fit(labels)
        res = self.le.transform(labels)
        self.oh = OneHotEncoder()
        self.oh.fit(np.reshape(res,(-1,1)))
        self.tmp_out = self.oh.transform(np.reshape(res,(-1,1)))

    def transform(self, y):
        tmp = self.le.transform(y)
        if DEBUG:
            print(tmp)
        res = self.oh.transform(np.reshape(tmp, (-1,1))).toarray()
        if DEBUG:
            print(res)
        return res

    def inverse_transform_v1(self, y):

        decode_columns = np.vectorize(lambda col: self.oh.active_features_[col])
        print("decode_columns:")
        print(decode_columns)


        print(np.shape(y))
        print(np.shape(self.tmp_out))

        decoded = decode_columns(y.indices).reshape(-1,np.shape(self.tmp_out)[-1])
        print("decoded")
        print(decoded)

        recovered_y = decoded - self.oh.feature_indices_[:-1]

        res = self.le.inverse_transform(recovered_y)
        print(res)
        return res

    def inverse_transform(self, y):

        decode_columns = np.vectorize(lambda col: self.oh.active_features_[col])
        print("decode_columns:")
        print(decode_columns)


        print(np.shape(y))
        print(np.shape(self.tmp_out))

        decoded = decode_columns(y.indices).reshape(-1,np.shape(self.tmp_out)[-1])
        print("decoded")
        print(decoded)

        recovered_y = decoded - self.oh.feature_indices_[:-1]

        res = self.le.inverse_transform(recovered_y)
        print(res)
        return res
 


def getIrisLabels(iris_data):
    # print(iris_data.groupby(4).sum())
    # print(iris_data[4].unique())
    return iris_data[4].unique()

def showIris(iris_data):
    print(iris_data.head(5))
    print(iris_data.columns)
    print(len(iris_data.columns))
    print(iris_data.index)

    print(iris_data[:,1])



    features = pd.DataFrame(iris_data, columns=[0,1,2,3])
    print(features.head(5))

    # print(features)

    labels = pd.DataFrame(iris_data, columns=[4])
    print(labels.head(5))

def traverse_iris():
    for i,row in iris_data.iterrows():
        if i>10:
            break
        # print(row)
        # print(row.values)
        # print(type(row.values))
        print(row.values[:4])
        print(row.values[4])



def eval_genomes_v1(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 4.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for i,row in iris_data.iterrows():
            output = net.activate(row.values[:4])
            # 这里需要改用cross entropy when task is classification
            # genome.fitness -= (output[0]-row[4])**2
            if output[0]!= row.values[4]:
                genome.fitness -= 1

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 4.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for i,row in iris_data.iterrows():
            output = net.activate(row.values[:4])
            # 这里需要改用cross entropy when task is classification
            # genome.fitness -= (output[0]-row[4])**2
            if DEBUG:
                print(output)
                print(row.values[4])

            label_encoded = zxxEncoder.transform([row.values[4]])
            # res = np.reshape(res, (-1))
            label_encoded = np.ravel(label_encoded).tolist()
            if DEBUG:
                print("label: {}".format(row.values[4]))
                print("after encoding: {}".format(label_encoded))

            soft_res = ZxxSoftmax(output)
            if DEBUG:
                print("soft_res: {}".format(soft_res))

            loss_cross = ZxxCrossEntropy(soft_res,label_encoded)
            loss_mse = MultiMSE(soft_res, label_encoded)

            # if output[0]!= row.values[4]:
            genome.fitness -= loss_cross


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
    winner = p.run(eval_genomes, GENS)

    #Display the winning genome.
    print('\n Best genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    predictNum = iris_data.shape[0]
    predictRight = 0.0
    for i,row in iris_data.iterrows():
        output = winner_net.activate(row.values[:4])
        index = output.index(max(output))
        expect_out = np.ravel(zxxEncoder.transform([row.values[4]]))
        expect_index = expect_out.tolist().index(max(expect_out))

        if index == expect_index:
            predictRight += 1

        # print('expected index {!r}, got {!r}     expected output {!r}, got {!r}   '.format(
        #     index, expect_index, expect_out, output
        # ))
    accuracy = predictRight/predictNum
    print("\nAccuracy: {}".format(accuracy), flush=True)

    # How does the node_names work?
    node_names = {-1:'A', -2:'B', -3:'C', -4:'D', 0:'label_0', 1:'label_1', 2:'label_2'}
    visualize.draw_net(config, winner, False, node_names=node_names)

    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    # p.run(eval_genomes, 1)

    return accuracy


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


def ZxxSoftmax(src):
    assert isinstance(src, list)
    assert len(src)>1
    denominator = sum([math.e ** i for i in src])
    return [(math.e ** i)/denominator for i in src]

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
        loss = -math.log(src[i])
        return loss
    except Exception as e:
        traceback.format_exc(e)
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

    iris_data = pd.read_csv("/Users/johnsaxon/test/github.com/learn-neat/meat/iris/testdata/iris_train.csv", header=None)

    labels = getIrisLabels(iris_data)

    # print("Iris labels: {}".format(labels))

    zxxEncoder = ZxxEncoder(labels)

    # traverse_iris()

    # Detemin path to configuration file.This path manipulation is
    # here so that the script will successfully regardless of the 
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config_feedforward')
    GENS=100
    while True:
        acc = run(config_path)
        if acc > 0.85 or GENS > 500:
            break
        GENS += 10

    # test_run(config_path)

    # getIrisLabels()
