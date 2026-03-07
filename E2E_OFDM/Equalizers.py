import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

# It's like declare the template of this 'block'. 
# Here says it is a 'Block' which has ONLY method (responsiblity) named 'process'.
class Equalizer(ABC):

    @abstractmethod
    def process(self, signal):
        ...

class ZeroForcing(Equalizer):
    def process(self, EstimatedChannel, ReceivedSignal):
        RecoveredSymbols = ReceivedSignal / EstimatedChannel
        return RecoveredSymbols
    
