from abc import ABC, abstractmethod
import numpy as np
from enum import Enum
from dataclasses import dataclass

# for choices of arguments, manually select instead of remember names of choices    
class CHANNEL_MODEL(Enum):
    RAYLEIGH = "RAYLEIGH"
    AWGN = "AWGN"
    NODISTORTION = "NODISTORTION"

@dataclass
class ChannelConfig:
    Model: CHANNEL_MODEL
    NumTap: int 
    RegenChannel: int       # Channels will change every n blocks

class Channel(ABC):
    Channels: np.ndarray
    Channels_feq: np.ndarray
    RegenChannel: int
    @abstractmethod
    def process(self, signal: np.ndarray, NumSubCarrier: int) -> np.ndarray:
        ...

class NoiseMixer():
    def __init__(self) -> None:
        self.VarNoises = []

    def process(self, signal: np.ndarray, SNR: float) -> np.ndarray:
        SNRlinear = 10 ** (SNR / 10)
        SignalPower = np.mean(np.abs(signal) ** 2, axis=0)
        NoisePower = SignalPower / SNRlinear
        Noise = np.dot((np.random.randn(signal.shape[0], signal.shape[1]) + 1j * np.random.randn(signal.shape[0], signal.shape[1])), np.diag(np.sqrt(NoisePower / 2)))
        self.Noise = Noise
        self.VarNoises.append(np.diag(NoisePower))
        ChannelOutput = signal + Noise
        return ChannelOutput

class RayleighChannel(Channel):
    def __init__(self, config):
        super().__init__()
        self.RegenChannel = config.RegenChannel
        self.NumTap = config.NumTap

    def process(self, signal: np.ndarray, NumSubCarrier: int) -> np.ndarray:
        Channels = []
        ChannelOutputs = []
        NumChannelRealization = int(np.ceil(signal.shape[0]/self.RegenChannel))
        for i in range(NumChannelRealization):
            channel = np.sqrt(1/2) * (np.random.randn(1,self.NumTap) + 1j * np.random.randn(1,self.NumTap))
            THISsignal = signal[i*self.RegenChannel:(i+1)*self.RegenChannel]
            Output = np.array([np.convolve(row, channel.reshape(-1), mode='full') for row in THISsignal])
            Channels.append(channel)
            ChannelOutputs.append(Output)
        Channels = np.concatenate(Channels, axis=0)
        Channels = np.concat((Channels, np.zeros((NumChannelRealization,NumSubCarrier-self.NumTap))), axis=1)
        self.Channels_feq = np.repeat(np.fft.fft(Channels), self.RegenChannel, axis=0)
        ChannelOutputs = np.concatenate(ChannelOutputs, axis=0)
        return ChannelOutputs
    
class AWGNChannel(Channel):
    # For AWGN channel, the noise corruption will be processed at the stage of mixing noise eventually.
    # So this acts like a no distortion channel first.
    def __init__(self):
        super().__init__()
        self.Channels = np.ndarray(1)

    def process(self, signal: np.ndarray) -> np.ndarray:
        ChannelOutput = signal
        return ChannelOutput
    
def SelectChannelModel(config: ChannelConfig) -> Channel:
    if config.Model == CHANNEL_MODEL.RAYLEIGH: 
        return RayleighChannel(config)
    elif config.Model == CHANNEL_MODEL.AWGN: 
        return AWGNChannel() 
    else: raise ValueError("Unsuport channel model")

if __name__ == '__main__':
    
    from dataclasses import dataclass
    
    @dataclass
    class AWGNChannelConfig():
        SNR: int = -6
        NumMC: int = 100000
        SeqLength: int = 10

    class TestAWGNChannel():
        def __init__(self, config) -> None:
            self.NumMC = config.NumMC
            self.SeqLength = config.SeqLength
            self.SNR = config.SNR
            self.NoiseMixer = NoiseMixer()

        def run(self):
            symbols = np.random.randn(self.NumMC, self.SeqLength) + np.random.randn(self.NumMC, self.SeqLength) * 1j
            NoisySymbols= self.NoiseMixer.process(symbols, self.SNR)
            NoisePower = np.mean(np.abs(self.NoiseMixer.Noise) ** 2, axis=0)
            SymbolPower = np.mean(np.abs(symbols) ** 2, axis=0)
            EstimatedSNR = 10 * np.log10(SymbolPower / NoisePower)
            SNRError = EstimatedSNR - self.SNR
            return SNRError

    config1 = AWGNChannelConfig()
    SNRError = TestAWGNChannel(config1).run()
    print("SNR Error: {}".format(SNRError))
