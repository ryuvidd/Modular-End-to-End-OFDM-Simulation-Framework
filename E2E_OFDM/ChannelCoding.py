import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum
import commpy.channelcoding.convcode as cc
from commpy.channelcoding import viterbi_decode

class ChannelEncoder(ABC):

    @abstractmethod
    def encode(self, bits: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def decode(self, bits: np.ndarray) -> np.ndarray:
        pass

class CHANNEL_ENCODER(Enum):
    Convolutional = "Convolutional Encoder"

class CONVOLUTIONAL_ENCODER_CHOICE(Enum):
    Kis3 = 'K = 3'
    Kis7 = 'K = 7'

@dataclass
class CONVOLUTIONAL_PARAM_SETTING:
    K: int
    taps: np.ndarray
    Rate: float
    memory: np.ndarray
    g_matrix: np.ndarray

class ConvolutionalCoding(ChannelEncoder):
    def __init__(self, choice: CONVOLUTIONAL_ENCODER_CHOICE, NumBits) -> None:
        ParamSetting = self._pick_setting(choice)
        self.K = ParamSetting.K
        self.taps = ParamSetting.taps
        self.register = np.zeros(self.K, dtype=np.int8)
        self.memory = ParamSetting.memory
        self.g_matrix = ParamSetting.g_matrix
        self.NumBits = NumBits

    def _pick_setting(self, choice: CONVOLUTIONAL_ENCODER_CHOICE) -> CONVOLUTIONAL_PARAM_SETTING:
        if choice == CONVOLUTIONAL_ENCODER_CHOICE.Kis3:
            setting = CONVOLUTIONAL_PARAM_SETTING(
                K = 3,
                taps = np.array([[1,1,1], [1,0,1]]),    # (7,5)
                Rate = 1/2,
                memory = np.array([2]),
                g_matrix = np.array([[7, 5]])
            )
        elif choice == CONVOLUTIONAL_ENCODER_CHOICE.Kis7:
            setting = CONVOLUTIONAL_PARAM_SETTING(
                K = 7,
                taps = np.array([[1,0,1,1,0,1,1], [1,1,1,1,0,0,1]]),    # (133,171)
                Rate = 1/2,
                memory = np.array([6]),
                g_matrix = np.array([[133, 171]])
            )
        else: 
            raise ValueError('Unsupported choice')
        return setting
    
    def encode_by_hand(self, bits: np.ndarray) -> np.ndarray:
        bits = bits.astype(np.int8)
        TailedBits = np.concatenate((bits, np.zeros(self.K-1, dtype=np.int8)))
        EncodedBits = []
        for bit in TailedBits:
            self.register[1:] = self.register[:-1]
            self.register[0] = bit

            outputs = (self.taps @ self.register) % 2
            EncodedBits.extend(outputs.tolist())
        EncodedBits = np.array(EncodedBits, dtype=np.int8)
        return EncodedBits
    
    def encode(self, bits: np.ndarray) -> np.ndarray:
        trellis = cc.Trellis(self.memory, self.g_matrix)
        EncodedBits = cc.conv_encode(bits.astype(int), trellis)
        EncodedBits = EncodedBits.astype(np.float64) 
        return EncodedBits
    
    def decode(self, EncodedBits: np.ndarray) -> np.ndarray:
        trellis = cc.Trellis(self.memory, self.g_matrix)
        EncodedBits = EncodedBits.astype(np.float64)
        DecodedBits = cc.viterbi_decode(
            EncodedBits, 
            trellis, 
            tb_depth=5, 
            decoding_type='hard'
        )
        DecodedBits = DecodedBits[:self.NumBits].astype(int)
        return DecodedBits

    
def SelectChannelEncoder(encoder: CHANNEL_ENCODER, choice: CONVOLUTIONAL_ENCODER_CHOICE, NumBits) -> ChannelEncoder:
    if encoder == CHANNEL_ENCODER.Convolutional: 
        return ConvolutionalCoding(choice, NumBits)
    else:
        raise ValueError("Unsupported channel encoder")

if __name__ == "__main__":
    encod = CHANNEL_ENCODER.Convolutional
    choice = CONVOLUTIONAL_ENCODER_CHOICE.Kis7

    N = 5000
    Block = SelectChannelEncoder(encod, choice, N)
    Bits = np.random.randint(0,2,N).astype(int)
    EncodedBits = Block.encode(Bits)
    EncodedBits = EncodedBits.astype(np.float64) 

    DecodedBits = Block.decode(EncodedBits)
    error = np.sum(Bits != DecodedBits)/N

    print(f"First 10 Bits:\t\t{Bits[:10]}")
    print(f"First 20 Encoded Bits:\t{EncodedBits[:20].astype(int)}")
    print(f"First 10 DecodedBits:\t{DecodedBits[:10]}")
    print(f"Total error: {error}")