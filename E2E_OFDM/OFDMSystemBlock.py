from dataclasses import dataclass
from Modulators import *
from Channels import *
from ChannelEstimators import *
from ChannelInterpolator import *
from EvaluationMetric import *
from BlocksGenerator import *
from Equalizers import *
from util import *

@dataclass
class OFDMconfig:
    NumSubCarrier: int
    TotalNumBlock: int
    NumPilotPerBlock: int
    QAMModulation: QAM_MODULATION
    LengthCP: int
    SNR: np.ndarray
    PilotTypes: PILOT_TYPES
    ChannelModel: CHANNEL_MODEL
    NumTap: int
    RegenChannel: int
    Estimator: CHANNEL_ESTIMATOR
    Interpolator: CHANNEL_INTERPOLATOR

class Transmitter():
    def __init__(self, config) -> None:
        self.BlocksGenerator = SelectPilotType(config)
        self.Modulator = SelectModulator(config.QAMModulation)
        self.OFDMModulator = OFDMModulator(config.LengthCP, config.NumSubCarrier)
        self.SpectrumViewer = SpectrumAnalyzer()

    def process(self, OFDMconfig) -> tuple:
        Blocks, DataBits = self.BlocksGenerator.process(self.Modulator.bitsPerSymbol)
        TransmittedSymbols = self.Modulator.modulate(Blocks)
        OFDMSymbols = self.OFDMModulator.modulate(TransmittedSymbols)
        self.SpectrumViewer.PlotPSD(OFDMSymbols)
        OFDMconfig.PilotIndices = self.BlocksGenerator.PilotIndices
        OFDMconfig.TransmittedSymbols = TransmittedSymbols
        return OFDMSymbols, DataBits, OFDMconfig
    
class ChannelBlock():
    def __init__(self, config) -> None:
        self.ChannelGenerator = SelectChannelModel(config)
        self.NoiseMixer = NoiseMixer()

    def process(self, OFDMSymbols: np.ndarray, OFDMconfig) -> tuple:
        ChannelOutput = self.ChannelGenerator.process(OFDMSymbols, OFDMconfig.NumSubCarrier)
        ReceivedSignals = []
        for snr in OFDMconfig.SNR:
            ReceivedSignals.append(self.NoiseMixer.process(ChannelOutput, snr))
        ReceivedSignals = np.stack(ReceivedSignals, axis=0)
        OFDMconfig.VarNoises = np.stack(self.NoiseMixer.VarNoises, axis=0)
        OFDMconfig.ChannelsFreqDomain = self.ChannelGenerator.Channels_feq
        return ReceivedSignals, OFDMconfig
    
class Receiver():
    def __init__(self, config) -> None:
        self.OFDMDemodulator = OFDMModulator(config.LengthCP, config.NumSubCarrier)
        self.Demodulator = SelectModulator(config.QAMModulation)
        self.ChannelEstimator = SelectEstimator(config.Estimator)
        self.ChannelInterpolator = SelectInterpolator(config.Interpolator)
        self.TotalNumBlock = config.TotalNumBlock
        self.Equalizer = ZeroForcing()
    
    def process(self, ReceivedSignals: np.ndarray, OFDMconfig) -> tuple:
        PilotIndices = OFDMconfig.PilotIndices
        EstimatedChannels = []
        EstimatedBits = []
        for snr_ind in range(len(OFDMconfig.SNR)):
            ReceivedSymbols = self.OFDMDemodulator.demodulate(ReceivedSignals[snr_ind])
            # Channel Estimation #
            ReceivedPilots = np.where(PilotIndices, ReceivedSymbols, 0)
            TransmittedPilots = np.where(PilotIndices, OFDMconfig.TransmittedSymbols, 0)
            metadata = {
                "VarNoise": OFDMconfig.VarNoises[snr_ind],
                "ChannelsFreqDomain": OFDMconfig.ChannelsFreqDomain,
                "LengthCP": OFDMconfig.LengthCP
            }
            EstimatedChannel = self.ChannelEstimator.process(ReceivedPilots, TransmittedPilots, metadata)
            EstimatedChannel = self.ChannelInterpolator.process(EstimatedChannel)
            EstimatedChannels.append(EstimatedChannel)
            # Equalization #
            RecoveredSymbols = self.Equalizer.process(EstimatedChannel, ReceivedSymbols)
            EstimatedDataSymbol = RecoveredSymbols[~PilotIndices].reshape(self.TotalNumBlock,-1)
            EstimatedBits_ = self.Demodulator.demodulate(EstimatedDataSymbol)
            EstimatedBits.append(EstimatedBits_)
        
        EstimatedChannels = np.stack(EstimatedChannels, axis=0)
        EstimatedBits = np.stack(EstimatedBits, axis=0)
        return EstimatedBits, EstimatedChannels

class OFDMSystem():
    def __init__(self, config) -> None:
        self.OFDMconfig = config
        self.Transmitter = Transmitter(config)
        self.ChannelBlock = ChannelBlock(config)
        self.Receiver = Receiver(config)
        self.Evaluater = TotalEvaluators(config.SNR)

    def run(self):
        OFDMSymbols, DataBits, config = self.Transmitter.process(self.OFDMconfig)
        logging.info("Transmitted signals")
        ReceivedSignals, config = self.ChannelBlock.process(OFDMSymbols, config)
        logging.info("Received signals and start recovering signal")
        EstimatedBits, EstimatedChannels = self.Receiver.process(ReceivedSignals, config)
        logging.info("Data recovered")

        # Duplicate for each SNR
        DataBits = np.tile(DataBits, (len(self.OFDMconfig.SNR),1,1))
        ChannelFreqDomain = np.tile(config.ChannelsFreqDomain, (len(self.OFDMconfig.SNR),1,1))

        EstimatedData = ExperimentData(
            Bits=EstimatedBits, 
            Channel=EstimatedChannels
        )
        GroundTruthData = ExperimentData(
            Bits=DataBits, 
            Channel=ChannelFreqDomain
        )
        results = self.Evaluater.process(EstimatedData, GroundTruthData)
        return results