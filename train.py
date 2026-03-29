# train.py
"""NEAT training using C++ cache data."""

import argparse
import pickle
from pathlib import Path

import neat

from config import Config
from cache_reader import load_cache
from fitness import evaluate_genome_racing, racing_theoretical_max


class GenomeEvaluator:
    def __init__(self, data, cfg, neat_config):
        self.data = data
        self.cfg = cfg
        self.neat_config = neat_config
        self.best_fitness = 0.0
        self.best_genome = None
        self.generation = 0
        self.theoretical_max = racing_theoretical_max(data)

    def eval_genomes(self, genomes, config):
        for genome_id, genome in genomes:
            net = neat.nn.RecurrentNetwork.create(genome, config)
            result = evaluate_genome_racing(net, self.data, self.cfg)

            # Apply parsimony penalty (matching C++)
            n_nodes = len(genome.nodes)
            n_conns = len(genome.connections)
            complexity = n_nodes + n_conns
            parsimony = 1.0 / (1.0 + 0.0002 * complexity)
            genome.fitness = result.fitness * parsimony

            if genome.fitness > self.best_fitness:
                self.best_fitness = genome.fitness
                self.best_genome = genome
                pct = 100.0 * result.frames_processed / max(1, result.total_frames)
                print(f"  [NEW BEST] fitness={genome.fitness:.1f} "
                      f"files={result.files_completed}/{result.total_files} "
                      f"acc={result.avg_accuracy:.3f} pct={pct:.1f}% "
                      f"nodes={n_nodes} conns={n_conns}")


def run_training(data_dir: str, save_path: str, config_path: str, generations: int):
    cfg = Config()
    cache_path = Path(data_dir) / "frames.cache"

    if not cache_path.exists():
        print(f"Cache not found at {cache_path}")
        print("Run your C++ program first to generate the cache.")
        return

    print(f"Network: {cfg.n_inputs} inputs, {cfg.n_outputs} outputs")
    data = load_cache(
        str(cache_path),
        midi_min=cfg.midi_min,
        midi_max=cfg.midi_max,
        hop_size=cfg.hop_size,
        sample_rate=cfg.sample_rate,
    )

    if not data:
        print("No recordings in cache!")
        return

    # Sort by name (matching C++ track order)
    data.sort(key=lambda r: r.name)
    print(f"\nTrack order ({len(data)} files):")
    for i, rec in enumerate(data):
        print(f"  {i+1}. {rec.name}")

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

    evaluator = GenomeEvaluator(data, cfg, neat_config)
    print(f"\nTheoretical max fitness: {evaluator.theoretical_max:.1f}")
    print(f"Population: {neat_config.pop_size}  Generations: {generations}\n")

    winner = pop.run(evaluator.eval_genomes, generations)

    with open(save_path, 'wb') as f:
        pickle.dump(winner, f)
    print(f"\nBest genome saved to {save_path}")
    print(f"Best fitness: {winner.fitness:.1f}")


def run_eval(genome_path: str, data_dir: str, config_path: str):
    cfg = Config()
    cache_path = Path(data_dir) / "frames.cache"

    with open(genome_path, 'rb') as f:
        genome = pickle.load(f)

    neat_config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    net = neat.nn.RecurrentNetwork.create(genome, neat_config)
    data = load_cache(str(cache_path), cfg.midi_min, cfg.midi_max, cfg.hop_size, cfg.sample_rate)
    data.sort(key=lambda r: r.name)

    from fitness import build_input

    total_tp = total_fp = total_fn = 0
    for rec in data:
        net.reset()
        tp = fp = fn = 0

        for fi in range(len(rec.frames)):
            inp = build_input(rec.frames, fi, cfg.pitch_history)
            out = net.activate(inp.tolist())

            for k in range(cfg.n_outputs):
                predicted = out[k] >= 0.5
                target = rec.frame_targets[fi, k] >= 0.5
                tp += predicted and target
                fp += predicted and not target
                fn += not predicted and target

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec_val / (prec + rec_val) if (prec + rec_val) > 0 else 0.0
        print(f"{rec.name:40s}  f1={f1:.4f}  prec={prec:.4f}  rec={rec_val:.4f}")

        total_tp += tp
        total_fp += fp
        total_fn += fn

    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    rec_val = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * prec * rec_val / (prec + rec_val) if (prec + rec_val) > 0 else 0.0
    print(f"\nOverall: f1={f1:.4f}  prec={prec:.4f}  rec={rec_val:.4f}")


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
