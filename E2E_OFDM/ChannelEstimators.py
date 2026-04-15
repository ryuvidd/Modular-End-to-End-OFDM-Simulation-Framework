from abc import ABC, abstractmethod
import numpy as np
from enum import Enum

# for choices of arguments, manually select instead of remember names of choices    
class CHANNEL_ESTIMATOR(Enum):
    LS = "LS"
    LMMSE = "LMMSE"

class Estimator(ABC):

    @abstractmethod
    def process(self, Rxsignal: np.ndarray, pilot: np.ndarray, metadata) -> np.ndarray:
        ...
    
class LSEstimator(Estimator):

    def process(self, RxSignal: np.ndarray, Pilot: np.ndarray, metadata) -> np.ndarray:
        
        # A = Pilot
        # temp = torch.linalg.pinv(torch.conj(A.T) @ A) @ torch.conj(A.T)
        # EstimatedChannel = temp @ RxSignal.to(torch.complex64)

        EstimatedChannel = np.divide(
            RxSignal,
            Pilot,
            out=np.full_like(RxSignal, np.nan, dtype=np.complex64),
            where=Pilot != 0
        )

        ## INTERPOLATION ##
        # x = np.arange(EstimatedChannel.shape[1])

        # for i in range(EstimatedChannel.shape[0]):
        #     row = EstimatedChannel[i]
        #     mask = ~np.isnan(row)

        #     EstimatedChannel[i].real = np.interp(x, x[mask], row.real[mask])
        #     EstimatedChannel[i].imag = np.interp(x, x[mask], row.imag[mask])

        return EstimatedChannel
    
class LMMSEEstimator(Estimator):
    def process(self, RxSignal: np.ndarray, Pilot: np.ndarray, metadata) -> np.ndarray:
        LengthCP = metadata["LengthCP"]
        VarNoise = metadata["VarNoise"]
        ActualChan = metadata["ChannelsFreqDomain"]
        ActualChan = ActualChan[:, :, np.newaxis]
        Pilot = Pilot[:, :, np.newaxis] * np.eye(Pilot.shape[1])

        mean_h = np.mean(ActualChan, axis=0)
        R_hh = np.mean(ActualChan @ np.transpose(ActualChan,(0,2,1)).conj(), axis=0)
        NoisePower = VarNoise[LengthCP:LengthCP+RxSignal.shape[1],LengthCP:LengthCP+RxSignal.shape[1]]

        EstimatedChannel = np.zeros_like(RxSignal, dtype=complex)
        for i in range(RxSignal.shape[0]):
            X = Pilot[i]
            y = RxSignal[i].reshape(-1,1)
            THISEstimatedChannel = mean_h + R_hh @ X.conj().T @ np.linalg.inv(X @ R_hh @ X.conj().T + NoisePower) @ (y - X @ mean_h)
            EstimatedChannel[i] = THISEstimatedChannel.reshape(-1)
        return EstimatedChannel


def SelectEstimator(estimator: CHANNEL_ESTIMATOR) -> Estimator:
    if estimator == CHANNEL_ESTIMATOR.LS: return LSEstimator()
    elif estimator == CHANNEL_ESTIMATOR.LMMSE: return LMMSEEstimator()
    else: raise ValueError("Unsuport estimator")