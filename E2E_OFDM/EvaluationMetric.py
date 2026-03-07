import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

@dataclass
class EvaluationResults:
    BER: Optional[float] = None
    ChannelNMSE: Optional[float] = None

@dataclass
class ExperimentData:
    Bits: np.ndarray
    Channel: np.ndarray

class Evaluator(ABC):
    @abstractmethod
    def process(self, signal: np.ndarray, target: np.ndarray) -> float:
        ...

class BER(Evaluator):
    def process(self, EstimatedBits: np.ndarray, Bits: np.ndarray) -> float:
        return float(np.mean(EstimatedBits != Bits))
    
class NMSE(Evaluator):
    def process(self, EstimatedSignal: np.ndarray, GroundTruthSignal: np.ndarray) -> float:
        numerator = np.mean(np.abs(EstimatedSignal - GroundTruthSignal) ** 2)
        denumerator = np.mean(np.abs(GroundTruthSignal) ** 2)
        NMSEdB = 10 * np.log10((numerator / denumerator))
        return float(NMSEdB)
    
class TotalEvaluators:
    def __init__(self):
        self.BEREvaluator = BER()
        self.ChannelEstimateEvaluator = NMSE()

    def process(self, Estimated: ExperimentData, GroundTruth: ExperimentData) -> EvaluationResults:
        results = EvaluationResults()
        results.BER = self.BEREvaluator.process(Estimated.Bits, GroundTruth.Bits)
        results.ChannelNMSE = self.ChannelEstimateEvaluator.process(Estimated.Channel, GroundTruth.Channel)
        return results
    
