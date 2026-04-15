import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class EvaluationResults:
    BER: np.ndarray
    ChannelNMSE: np.ndarray

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
    def __init__(self, SNR):
        self.BEREvaluator = BER()
        self.ChannelEstimateEvaluator = NMSE()
        self.SNR = SNR

    def process(self, Estimated: ExperimentData, GroundTruth: ExperimentData) -> EvaluationResults:
        
        TotalBER = []
        TotalChannelNMSE = []
        for snr_ind in range(len(self.SNR)):
            BER_ = self.BEREvaluator.process(Estimated.Bits[snr_ind], GroundTruth.Bits[snr_ind])
            ChannelNMSE_ = self.ChannelEstimateEvaluator.process(Estimated.Channel[snr_ind], GroundTruth.Channel[snr_ind])
            
            TotalBER.append(BER_)
            TotalChannelNMSE.append(ChannelNMSE_)
        
        results = EvaluationResults(
            BER=np.array(TotalBER),
            ChannelNMSE=np.array(TotalChannelNMSE)
        )
        return results
    
