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
    Estimator: CHANNEL_ESTIMATOR
    Interpolator: CHANNEL_INTERPOLATOR

    # Channel Configuration 
    ChannelModel: CHANNEL_MODEL
    NumTap: int
    RegenChannel: int


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

        metadata = {}
        metadata["PilotIndices"] = self.BlocksGenerator.PilotIndices
        metadata["TransmittedSymbols"] = TransmittedSymbols
        metadata["DataBits"] = DataBits
        return OFDMSymbols, OFDMconfig, metadata
    
class ChannelBlock():
    def __init__(self, config) -> None:
        self.ChannelGenerator = SelectChannelModel(config)
        self.NoiseMixer = NoiseMixer()

    def process(self, OFDMSymbols: np.ndarray, OFDMconfig, metadata) -> tuple:
        ChannelOutput = self.ChannelGenerator.process(OFDMSymbols, OFDMconfig.NumSubCarrier)
        ReceivedSignals = []
        for snr in OFDMconfig.SNR:
            ReceivedSignals.append(self.NoiseMixer.process(ChannelOutput, snr))
        ReceivedSignals = np.stack(ReceivedSignals, axis=0)

        metadata["VarNoises"] = np.stack(self.NoiseMixer.VarNoises, axis=0)
        metadata["ChannelsFreqDomain"] = self.ChannelGenerator.Channels_feq
        return ReceivedSignals, OFDMconfig, metadata
    
class Receiver():
    def __init__(self, config) -> None:
        self.OFDMDemodulator = OFDMModulator(config.LengthCP, config.NumSubCarrier)
        self.Demodulator = SelectModulator(config.QAMModulation)
        self.ChannelEstimator = SelectEstimator(config.Estimator)
        self.ChannelInterpolator = SelectInterpolator(config.Interpolator)
        self.TotalNumBlock = config.TotalNumBlock
        self.Equalizer = ZeroForcing()
    
    def process(self, ReceivedSignals: np.ndarray, OFDMconfig, metadata) -> tuple:
        PilotIndices = metadata["PilotIndices"]
        EstimatedChannels = []
        EstimatedBits = []
        for snr_ind in range(len(OFDMconfig.SNR)):
            ReceivedSymbols = self.OFDMDemodulator.demodulate(ReceivedSignals[snr_ind])
            # Channel Estimation #
            ReceivedPilots = np.where(PilotIndices, ReceivedSymbols, 0)
            TransmittedPilots = np.where(PilotIndices, metadata["TransmittedSymbols"], 0)
            EstimationMetadata = {
                "VarNoise": metadata["VarNoises"][snr_ind],
                "ChannelsFreqDomain": metadata["ChannelsFreqDomain"],
                "LengthCP": OFDMconfig.LengthCP
            }
            EstimatedChannel = self.ChannelEstimator.process(ReceivedPilots, TransmittedPilots, EstimationMetadata)
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
        OFDMSymbols, config, metadata = self.Transmitter.process(self.OFDMconfig)
        logging.info("Transmitted signals")
        ReceivedSignals, config, metadata = self.ChannelBlock.process(OFDMSymbols, config, metadata)
        logging.info("Received signals and start recovering signal")
        EstimatedBits, EstimatedChannels = self.Receiver.process(ReceivedSignals, config, metadata)
        logging.info("Data recovered")

        # Duplicate for each SNR
        DataBits = np.tile(metadata["DataBits"], (len(self.OFDMconfig.SNR),1,1))
        ChannelFreqDomain = np.tile(metadata["ChannelsFreqDomain"], (len(self.OFDMconfig.SNR),1,1))

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