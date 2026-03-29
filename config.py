# config.py
"""NEAT and audio configuration."""

from dataclasses import dataclass


@dataclass
class Config:
    # Population
    pop_size: int = 300
    generations: int = 500

    # Audio (must match C++ NeatConfig)
    n_cqt_bins: int = 108
    n_salience_bins: int = 49
    hop_size: int = 512
    sample_rate: int = 22050
    pitch_history: int = 8

    # MIDI range
    midi_min: int = 40
    midi_max: int = 88

    @property
    def n_inputs(self) -> int:
        return self.n_cqt_bins + self.n_salience_bins + self.pitch_history

    @property
    def n_outputs(self) -> int:
        return self.midi_max - self.midi_min + 1
