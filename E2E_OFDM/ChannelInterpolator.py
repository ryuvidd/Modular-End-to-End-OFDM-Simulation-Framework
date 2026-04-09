import numpy as np
from abc import ABC, abstractmethod
from enum import Enum

class CHANNEL_INTERPOLATOR(Enum):
    LINEAR = "LINEAR"

class Interpolator(ABC):

    @abstractmethod
    def process(self, EstimatedChannel: np.ndarray) -> np.ndarray:
        ...

class LinearInterpolator(Interpolator):
    def process(self, EstimatedChannel: np.ndarray) -> np.ndarray:
        x = np.arange(EstimatedChannel.shape[1])

        for i in range(EstimatedChannel.shape[0]):
            row = EstimatedChannel[i]
            mask = ~np.isnan(row)

            EstimatedChannel[i].real = np.interp(x, x[mask], row.real[mask])
            EstimatedChannel[i].imag = np.interp(x, x[mask], row.imag[mask])
        return EstimatedChannel
    
def SelectInterpolator(interpolator: CHANNEL_INTERPOLATOR) -> Interpolator:
    if interpolator == CHANNEL_INTERPOLATOR.LINEAR: return LinearInterpolator()
    else: raise ValueError("Unsupported interpolator")

