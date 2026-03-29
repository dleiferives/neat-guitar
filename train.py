# train.py
"""NEAT training using C++ fitness evaluation."""

import argparse
import pickle
from pathlib import Path

import neat

import neat_fitness


class GenomeEvaluator:
    def __init__(self):
        self.best_fitness = 0.0
        self.best_genome = None
        self.generation = 0
        self.lookback = 0
        self.stagnation = 0
        self.theoretical_max = neat_fitness.get_theoretical_max()

    def _genome_to_cpp(self, genome):
        """Convert neat-python genome to C++ format."""
        # nodes: (key, bias, response)
        nodes = []
        for key, node in genome.nodes.items():
            nodes.append((key, node.bias, node.response))

        # conns: (in_node, out_node, weight, enabled)
        conns = []
        for (in_node, out_node), conn in genome.connections.items():
            conns.append((in_node, out_node, conn.weight, conn.enabled))

        return nodes, conns

    def eval_genomes(self, genomes, config):
        for genome_id, genome in genomes:
            nodes, conns = self._genome_to_cpp(genome)

            # Evaluate from start position 0
            result = neat_fitness.evaluate_genome(nodes, conns, 0)
            fitness = result['fitness']

            # Multi-start if stagnating
            for lb in range(1, self.lookback + 1):
                n_files = neat_fitness.get_num_recordings()
                alt_start = n_files - lb
                if alt_start > 0:
                    alt_result = neat_fitness.evaluate_genome(nodes, conns, alt_start)
                    fitness += alt_result['fitness']

            genome.fitness = fitness

            if genome.fitness > self.best_fitness:
                self.best_fitness = genome.fitness
                self.best_genome = genome
                self.stagnation = 0

                pct = 100.0 * result['frames_processed'] / max(1, result['total_frames'])
                print(f"  [NEW BEST] fitness={genome.fitness:.1f} "
                      f"files={result['files_completed']}/{result['total_files']} "
                      f"acc={result['avg_accuracy']:.3f} pct={pct:.1f}% "
                      f"nodes={result['n_nodes']} conns={result['n_conns']}")

        self.stagnation += 1

        # Progressive multi-start (matching C++ logic)
        if self.stagnation >= 100:
            new_lookback = (self.stagnation - 100) // 100 + 1
            n_files = neat_fitness.get_num_recordings()
            new_lookback = min(new_lookback, n_files - 1)
            if new_lookback > self.lookback:
                self.lookback = new_lookback
                print(f"\n  [MULTI-START] now combining {self.lookback + 1} "
                      f"starting positions (stag={self.stagnation})")
                self.theoretical_max = neat_fitness.get_theoretical_max() * (self.lookback + 1)
                self.stagnation = 0
                self.best_fitness = 0.0


def run_training(data_dir: str, save_path: str, config_path: str, generations: int):
    print(f"Loading data from {data_dir}...")
    if not neat_fitness.load_data(data_dir):
        print("Failed to load data!")
        return

    n_inputs = neat_fitness.get_n_inputs()
    n_outputs = neat_fitness.get_n_outputs()
    print(f"Network: {n_inputs} inputs, {n_outputs} outputs")

    names = neat_fitness.get_recording_names()
    print(f"\nTrack order ({len(names)} files):")
    for i, name in enumerate(names):
        print(f"  {i+1}. {name}")

    neat_config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    pop = neat.Population(neat_config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.Checkpointer(50, filename_prefix='neat-checkpoint-'))

    evaluator = GenomeEvaluator()
    print(f"\nTheoretical max fitness: {evaluator.theoretical_max:.1f}")
    print(f"Population: {neat_config.pop_size}  Generations: {generations}\n")

    winner = pop.run(evaluator.eval_genomes, generations)

    with open(save_path, 'wb') as f:
        pickle.dump(winner, f)
    print(f"\nBest genome saved to {save_path}")
    print(f"Best fitness: {winner.fitness:.1f}")


def run_eval(genome_path: str, data_dir: str, config_path: str):
    if not neat_fitness.load_data(data_dir):
        print("Failed to load data!")
        return

    with open(genome_path, 'rb') as f:
        genome = pickle.load(f)

    nodes = [(k, n.bias, n.response) for k, n in genome.nodes.items()]
    conns = [(c.key[0], c.key[1], c.weight, c.enabled)
             for c in genome.connections.values()]

    result = neat_fitness.evaluate_genome(nodes, conns)
    print(f"Fitness: {result['fitness']:.1f}")
    print(f"Files completed: {result['files_completed']}/{result['total_files']}")
    print(f"Frames: {result['frames_processed']}/{result['total_frames']}")
    print(f"Avg accuracy: {result['avg_accuracy']:.4f}")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', required=True)

    train_p = subparsers.add_parser('train')
    train_p.add_argument('data_dir')
    train_p.add_argument('--save', default='best_genome.pkl')
    train_p.add_argument('--config', default='neat_config.ini')
    train_p.add_argument('--generations', type=int, default=500)

    eval_p = subparsers.add_parser('eval')
    eval_p.add_argument('genome')
    eval_p.add_argument('data_dir')
    eval_p.add_argument('--config', default='neat_config.ini')

    args = parser.parse_args()

    if args.command == 'train':
        run_training(args.data_dir, args.save, args.config, args.generations)
    elif args.command == 'eval':
        run_eval(args.genome, args.data_dir, args.config)


if __name__ == '__main__':
    main()
