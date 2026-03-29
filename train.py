# train.py
"""Main training script using neat-python."""

import argparse
import os
import pickle
import sys
from pathlib import Path

import neat
import numpy as np

from config import Config
from dataset import load_recordings
from fitness import evaluate_genome_racing, RacingResult


class GenomeEvaluator:
    """Evaluates genomes using racing fitness."""

    def __init__(self, data, cfg, neat_config):
        self.data = data
        self.cfg = cfg
        self.neat_config = neat_config
        self.best_fitness = 0.0
        self.best_genome = None
        self.generation = 0

    def eval_genomes(self, genomes, config):
        """Evaluate all genomes in the population."""
        for genome_id, genome in genomes:
            net = neat.nn.RecurrentNetwork.create(genome, config)
            result = evaluate_genome_racing(net, self.data, self.cfg)
            genome.fitness = result.fitness

            if genome.fitness > self.best_fitness:
                self.best_fitness = genome.fitness
                self.best_genome = genome
                print(f"  [NEW BEST] fitness={genome.fitness:.1f} "
                      f"files={result.files_completed}/{result.total_files} "
                      f"acc={result.avg_accuracy:.3f}")


def run_training(data_dir: str, save_path: str, config_path: str, generations: int):
    """Run NEAT training."""
    cfg = Config()

    # Update num_inputs in config file if needed
    print(f"Network: {cfg.n_inputs} inputs, {cfg.n_outputs} outputs")

    # Load data
    print(f"Loading recordings from {data_dir}...")
    data = load_recordings(data_dir, cfg)
    if not data:
        print("No recordings found!")
        return

    # Sort by name for consistent track order
    data.sort(key=lambda r: r.name)
    print(f"\nTrack order ({len(data)} files):")
    for i, rec in enumerate(data):
        print(f"  {i+1}. {rec.name}")

    # Load NEAT config
    neat_config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    # Create population
    pop = neat.Population(neat_config)

    # Add reporters
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.Checkpointer(
        generation_interval=50,
        filename_prefix='neat-checkpoint-'
    ))

    # Create evaluator
    evaluator = GenomeEvaluator(data, cfg, neat_config)

    # Run evolution
    winner = pop.run(evaluator.eval_genomes, generations)

    # Save best genome
    with open(save_path, 'wb') as f:
        pickle.dump(winner, f)
    print(f"\nBest genome saved to {save_path}")
    print(f"Best fitness: {winner.fitness:.1f}")

    return winner


def run_eval(genome_path: str, data_dir: str, config_path: str):
    """Evaluate a saved genome."""
    cfg = Config()

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

    data = load_recordings(data_dir, cfg)
    data.sort(key=lambda r: r.name)

    # Per-file evaluation
    total_tp = total_fp = total_fn = 0
    for rec in data:
        net.reset()
        tp = fp = fn = 0

        for fi in range(len(rec.frames)):
            from processing import build_input
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

    # Overall
    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    rec_val = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * prec * rec_val / (prec + rec_val) if (prec + rec_val) > 0 else 0.0
    print(f"\nOverall: f1={f1:.4f}  prec={prec:.4f}  rec={rec_val:.4f}")


def main():
    parser = argparse.ArgumentParser(description='NEAT Audio Transcription')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Train
    train_p = subparsers.add_parser('train')
    train_p.add_argument('data_dir', help='Recordings directory')
    train_p.add_argument('--save', default='best_genome.pkl')
    train_p.add_argument('--config', default='neat_config.ini')
    train_p.add_argument('--generations', type=int, default=500)

    # Eval
    eval_p = subparsers.add_parser('eval')
    eval_p.add_argument('genome', help='Saved genome file')
    eval_p.add_argument('data_dir', help='Recordings directory')
    eval_p.add_argument('--config', default='neat_config.ini')

    args = parser.parse_args()

    if args.command == 'train':
        run_training(args.data_dir, args.save, args.config, args.generations)
    elif args.command == 'eval':
        run_eval(args.genome, args.data_dir, args.config)


if __name__ == '__main__':
    main()
