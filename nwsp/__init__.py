"""NWSP: Neural Wasserstein Splines for trajectory inference from population snapshots."""
import logging

from .model import CubicBSpline, SplineGenerator
from .potentials import Dictionary, NeuralPotentialBlock, build_dictionary
from .trainer import NWSPConfig, NWSPNConfig, NWSPNTrainer, NWSPTrainer, load_model

__version__ = "1.0.0"
__all__ = ["CubicBSpline", "Dictionary", "NWSPConfig", "NWSPNConfig", "NWSPNTrainer", "NWSPTrainer",
           "NeuralPotentialBlock", "SplineGenerator", "__version__", "build_dictionary", "load_model"]

logging.getLogger(__name__).addHandler(logging.NullHandler())
