import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

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
    
class SpectrumAnalyzer():
    def PlotPSD(self, OFDMsignal):
        N = OFDMsignal.shape[1]
        rectangular_window = np.ones(N)
        n_fft = 1024
        freq = np.linspace(-0.5, 0.5, n_fft)

        WindowedSignal = OFDMsignal * rectangular_window
        WindowedSignal = WindowedSignal / np.sqrt(np.mean(rectangular_window**2))
        FreqSignal = np.fft.fft(WindowedSignal, n=n_fft)
        FreqSignal = np.fft.fftshift(FreqSignal, axes=1)
        PSD = (1/N) * np.abs(FreqSignal)**2
        PSD_avg = np.mean(PSD, axis=0)
        PSD_dB = 10 * np.log10(PSD_avg + 1e-12)

        plt.figure()
        plt.plot(freq, PSD_dB)
        plt.xlabel('Normalized Frequencies')
        plt.grid()
        plt.ylabel('Magnitude of Spectrum (dB)')
        plt.title('Power Spectral Density')
        
        plt.tight_layout()
        plt.savefig('results/PSD_transmitted_signal.png', dpi=300)

