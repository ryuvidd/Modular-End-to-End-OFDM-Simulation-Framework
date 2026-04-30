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
        NumInterleavedBits = metadata["NumInterleavedBits"]
        NumCol = metadata["NumCol"]
        NumPaddedZeros = metadata["NumPaddedZeros"]

        DeinterleavingShape = (NumCol, NumCol)
        signal = signal.flatten()[:NumInterleavedBits]
        matrix = np.reshape(signal.flatten(), DeinterleavingShape)
        DeinterleavedSignal = matrix.T.flatten()[:-NumPaddedZeros] if NumPaddedZeros !=0 else matrix.T.flatten()
        return DeinterleavedSignal
    
if __name__ == "__main__":
    from EvaluationMetric import *

    Interleaver = InterleaverBlock()

    N = 10000
    Bits = np.random.randint(0,2,N)
    InterleavedBits, NumCol, NumPaddedZeros = Interleaver.interleave(Bits)
    metadata = {
        "InterleavingShape": NumCol,
        "InterleavingZeros": NumPaddedZeros
    }
    DeinterleavedBits = Interleaver.deinterleave(InterleavedBits, metadata)

    Eval = BER()
    BER_ = Eval.process(DeinterleavedBits, Bits)
    print(f"BER: {BER_}")