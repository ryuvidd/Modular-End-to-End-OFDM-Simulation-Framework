import numpy as np
from abc import ABC, abstractmethod
from enum import Enum

class EQUALIZER(Enum):
    ZeroForcing = "ZeroForcing"
    MMSE = "MMSE"

class Equalizer(ABC):

    @abstractmethod
    def process(self, EstimatedChannel: np.ndarray, ReceivedSignal: np.ndarray) -> np.ndarray:
        ...

class ZeroForcing(Equalizer):
    def process(self, EstimatedChannel, ReceivedSignal):
        RecoveredSymbols = ReceivedSignal / EstimatedChannel
        return RecoveredSymbols
    
class MMSE(Equalizer):
    def process(self, EstimatedChannel, ReceivedSignal):
        RecoveredSymbols = ReceivedSignal / EstimatedChannel # to be fixed soon
        return RecoveredSymbols
    
def SelectEqualizer(equalizer: EQUALIZER) -> Equalizer:
    if equalizer == EQUALIZER.ZeroForcing: return ZeroForcing()
    elif equalizer == EQUALIZER.MMSE: return MMSE()
    else: raise ValueError("Unsuport equalizer")
    
