import os
import neat
import visualize
import numpy as np
import gym

env = gym.make('CartPole-v0').unwrapped
max_episode = 10
winner_max_episode = 10
episode_step = 50  # control the max steps of each episode
is_training = False  # True mean on-evolving, False mean output the best network
checkpoint = 9  # final state

# the reward is 1.0 if the pole does not fall down.
# fitness depends on the total reward of each episode.
# the minimum of the eposode is the fitness


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        episode_reward = []
        for episode in range(max_episode):
            accumulative_reward = 0
            observation = env.resst()
            for step in range(episode_step):
                action = np.argmax(net.activate(observation))
                observation_, reward, done, _ = env.step(action)
                accumulative_reward += reward
                observation = observation_
                if done:
                    break
            episode_reward.append(accumulative_reward)
        genome.fitness = np.min(episode_reward)/episode_step


def run(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))
    p.run(eval_genomes, 10)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)


def evoluation():
    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-%d' % checkpoint)
    winner = p.run(eval_genomes, 1)
    net = neat.nn.FeedForwardNetwork.create(winner, p.config)
    for episode in range(winner_max_episode):
        s = env.reset()
        for step in range(100):
            env.render()
            a = np.argmax(net.activate(s))
            s, r, done, _ = env.step(a)
            if done:
                break
    node_names = {-1: 'x', -2: 'x_dot', -3: 'theta', -
                  4: 'theta_dot', 0: 'action1', 1: 'action2'}
    visualize.draw_net(p.config, winner, view=True, node_names=node_names)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-forward')
    if is_training is True:
        run(config_path)
    else:
        evoluation()
