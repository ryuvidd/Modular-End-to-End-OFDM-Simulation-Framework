import numpy as np

class InterleaverBlock():

    def _Find_Interleaving_Shape(self, SignalSize: int) -> tuple: 
        NumCol = int(np.ceil(np.sqrt(SignalSize)))
        NumCol = NumCol if NumCol%2==0 else NumCol+1
        NumPaddedZeros = NumCol ** 2 - SignalSize
        return NumCol, NumPaddedZeros

    def interleave(self, signal: np.ndarray) -> tuple:
        NumCol, NumPaddedZeros = self._Find_Interleaving_Shape(signal.size)
        signal = np.concatenate((signal, np.zeros(NumPaddedZeros)))
        matrix = np.reshape(signal, (NumCol, NumCol))
        InterleavedSignal = matrix.T.flatten()
        return InterleavedSignal, NumCol, NumPaddedZeros
    
    def deinterleave(self, signal: np.ndarray, metadata) -> np.ndarray:
        NumCol = metadata["InterleavingShape"]
        NumPaddedZeros = metadata["InterleavingZeros"]
        DeinterleavingShape = (NumCol, NumCol)
        matrix = np.reshape(signal.flatten(), DeinterleavingShape)
        DeinterleavedSignal = matrix.T.flatten()[:-NumPaddedZeros]
        return DeinterleavedSignal