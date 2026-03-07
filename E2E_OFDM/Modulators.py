from abc import ABC, abstractmethod
import numpy as np
from enum import Enum

"""
For Demodulation: There would be 
1) Hard decision: minimum distance
2) Slicing: nearest imag and real (works for square QAM)
3) Soft decision: LLR, Max-Log-MAP

Here just 2) Slicing first
"""
# for choices of arguments, manually select instead of remember names of choices    
class QAM_MODULATION(Enum):
    QPSK_GRAY = "QPSK_GRAY"
    BPSK = "BPSK"

class Modulator(ABC):
    bitsPerSymbol: int

    @abstractmethod
    def modulate(self, bits: np.ndarray) -> np.ndarray:
        ...
    
    def demodulate(self, symbols: np.ndarray) -> np.ndarray:
        ...

class QPSKGrayCodedModulator(Modulator):
    def __init__(self):
        super().__init__()
        self.bitsPerSymbol = 2

    def modulate(self, bits: np.ndarray) -> np.ndarray:
        ModulatedSymbols = (2 * bits[:,::2] - 1) + 1j * (2 * bits[:,1::2] - 1)
        return ModulatedSymbols
    
    def demodulate(self, symbols: np.ndarray) -> np.ndarray:
        
        EstimatedSymbols = np.sign(np.real(symbols)) + 1j*(np.sign(np.imag(symbols)))

        EstimatedBits = np.empty((EstimatedSymbols.shape[0], EstimatedSymbols.shape[1]*2), dtype=int) 
        EstimatedBits[:,0::2] = (np.real(EstimatedSymbols) + 1) / 2
        EstimatedBits[:,1::2] = (np.imag(EstimatedSymbols) + 1) / 2
        return EstimatedBits

class BPSKModulator(Modulator):
    def __init__(self):
        super().__init__()
        self.bitsPerSymbol = 1

    def modulate(self, bits: np.ndarray) -> np.ndarray:
        ModulatedSymbols = 2 * bits - 1
        return ModulatedSymbols
    
    def demodulate(self, symbols: np.ndarray) -> np.ndarray:
        EstimatedBits = (symbols + 1) / 2
        return EstimatedBits
    
def SelectModulator(modulation_type: QAM_MODULATION) -> Modulator:
    if modulation_type == QAM_MODULATION.QPSK_GRAY: return QPSKGrayCodedModulator()
    elif modulation_type == QAM_MODULATION.BPSK: return BPSKModulator()
    else: raise ValueError("Unsupported modulation")

class OFDMModulator(Modulator):
    def __init__(self, LengthCP):
        super().__init__()
        self.LengthCP = LengthCP

    def modulate(self, symbols: np.ndarray) -> np.ndarray:
        IFFTSequence = np.fft.ifft(symbols, norm='ortho')  # IFFT = modulate each symbols with orthogonal frequencies and sum
        self.SequenceLength = IFFTSequence.shape[1]
        CP = IFFTSequence[:,-self.LengthCP:]
        OFDMSymbols = np.concatenate([CP, IFFTSequence], axis=1)
        return OFDMSymbols
    
    def demodulate(self, OFDMSymbols: np.ndarray) -> np.ndarray:
        CPR = OFDMSymbols[:,self.LengthCP:self.SequenceLength+self.LengthCP]
        DemodulatedSymbols = np.fft.fft(CPR, norm='ortho')
        return DemodulatedSymbols

if __name__ == '__main__':

    from dataclasses import dataclass
    from Channels import *
    from EvaluationMetric import *

    @dataclass
    class ModulatorConfig:
        NumBits: int = 100000
        ModulatorType: str = 'QPSKGreyCoded'
        SNR: int = 10

    class TestModulator():
        def __init__(self, config) -> None:
            self.modulator = SelectModulator(config.ModulatorType)
            self.Noise = AWGNChannel(config.SNR)
            self.Evaluater = BER()
        
        def run(self):
            bits = np.random.randint(0, 2, config.NumBits)
            ModulatedSymbols = self.modulator.modulate(bits)
            NoisySymbols = self.Noise.process(ModulatedSymbols, ModulatedSymbols.shape[1])
            EstimatedBits = self.modulator.demodulate(NoisySymbols)
            BERResult = self.Evaluater.process(EstimatedBits, bits)
            return BERResult
            
    config = ModulatorConfig()
    BERResult = TestModulator(config).run()
    print("BER: {}".format(BERResult))
    

    



