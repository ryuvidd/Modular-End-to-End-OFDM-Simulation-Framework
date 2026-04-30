from dataclasses import dataclass
from Modulators import *
from Channels import *
from ChannelEstimators import *
from ChannelInterpolator import *
from EvaluationMetric import *
from BlocksGenerator import *
from Equalizers import *
from ChannelCoding import *
from Interleaver import *
from util import *

@dataclass
class OFDMconfig:
    NumSubCarrier: int
    NumBits: int
    NumPilotPerBlock: int
    QAMModulation: QAM_MODULATION
    LengthCP: int
    SNR: np.ndarray
    PilotTypes: PILOT_TYPES
    Estimator: CHANNEL_ESTIMATOR
    Interpolator: CHANNEL_INTERPOLATOR
    Equalizer: EQUALIZER
    ChannelCodingType: CHANNEL_ENCODER
    ChannelCodingSetting: CONVOLUTIONAL_ENCODER_CHOICE

    # Channel Configuration 
    ChannelModel: CHANNEL_MODEL
    NumTap: int
    RegenChannel: int


class Transmitter():
    def __init__(self, config) -> None:
        self.BitGenerator = BitsGenerator(config.NumBits)
        self.ChannelEncoder = SelectChannelEncoder(config.ChannelCodingType, config.ChannelCodingSetting, config.NumBits)
        self.Interleaver = InterleaverBlock()
        self.Modulator = SelectModulator(config.QAMModulation)
        self.PilotInsertion = SelectPilotType(config)
        self.OFDMModulator = OFDMModulator(config.LengthCP, config.NumSubCarrier)
        self.SpectrumViewer = SpectrumAnalyzer()

    def process(self) -> tuple:
        DataBits = self.BitGenerator.process()
        CodedDataBits = self.ChannelEncoder.encode(DataBits)
        InterleavedBits, NumCol, NumPaddedZeros = self.Interleaver.interleave(CodedDataBits)
        DataSymbols = self.Modulator.modulate(InterleavedBits)
        Blocks, NumExtraBits, PilotIndices = self.PilotInsertion.process(DataSymbols, self.Modulator.bitsPerSymbol)
        OFDMSymbols = self.OFDMModulator.modulate(Blocks)
        self.SpectrumViewer.PlotPSD(OFDMSymbols)

        metadata = {}
        metadata["PilotIndices"] = PilotIndices
        metadata["NumExtraBits"] = NumExtraBits
        metadata["Blocks"] = Blocks
        metadata["DataBits"] = DataBits
        metadata["DataBitSize"] = DataBits.size
        metadata["InterleavingShape"] = NumCol
        metadata["InterleavingZeros"] = NumPaddedZeros
        return OFDMSymbols, metadata
    
class ChannelBlock():
    def __init__(self, config) -> None:
        self.ChannelGenerator = SelectChannelModel(config)
        self.NumSubCarrier = config.NumSubCarrier
        self.NumTap = config.NumTap
        self.RegenChannel = config.RegenChannel
        self.NoiseMixer = NoiseMixer()
        self.SNRs = config.SNR

    def process(self, OFDMSymbols: np.ndarray, metadata) -> tuple:
        NumBlocks = OFDMSymbols.shape[0]
        NumChannelRealization = int(np.ceil(NumBlocks/self.RegenChannel))
        ChannelRealizations = np.zeros((NumChannelRealization, self.NumSubCarrier), dtype=complex)

        ChannelOutput, Channels = self.ChannelGenerator.process(OFDMSymbols, NumChannelRealization)
        ChannelRealizations[:,:self.NumTap] = Channels
        Channels_freq = np.repeat(np.fft.fft(ChannelRealizations), self.RegenChannel, axis=0)[:NumBlocks]

        ReceivedSignals = []
        for snr in self.SNRs:
            ReceivedSignals.append(self.NoiseMixer.process(ChannelOutput, snr))
        ReceivedSignals = np.stack(ReceivedSignals, axis=0)

        metadata["NumBlocks"] = NumBlocks
        metadata["VarNoises"] = np.stack(self.NoiseMixer.VarNoises, axis=0)
        metadata["ChannelsFreqDomain"] = Channels_freq
        return ReceivedSignals, metadata
    
class Receiver():
    def __init__(self, config) -> None:
        self.OFDMDemodulator = OFDMModulator(config.LengthCP, config.NumSubCarrier)
        self.Demodulator = SelectModulator(config.QAMModulation)
        self.ChannelEstimator = SelectEstimator(config.Estimator)
        self.ChannelInterpolator = SelectInterpolator(config.Interpolator)
        self.Equalizer = SelectEqualizer(config.Equalizer)
        self.Deinterleaver = InterleaverBlock()
        self.ChannelDecoder = SelectChannelEncoder(config.ChannelCodingType, config.ChannelCodingSetting, config.NumBits)
        self.SNRs = config.SNR
        self.LengthCP = config.LengthCP
        self.NumSubCarrier = config.NumSubCarrier
    
    def process(self, ReceivedSignals: np.ndarray, metadata) -> tuple:
        PilotIndices = metadata["PilotIndices"]
        EstimatedChannels = []
        EstimatedBits = []

        ActualChan = metadata["ChannelsFreqDomain"][:, :, np.newaxis]
        mean_h = np.mean(ActualChan, axis=0)
        R_hh = np.mean(ActualChan @ np.transpose(ActualChan,(0,2,1)).conj(), axis=0)

        for snr_ind in range(len(self.SNRs)):
            ReceivedSymbols = self.OFDMDemodulator.demodulate(ReceivedSignals[snr_ind])
            # Channel Estimation #
            ReceivedPilots = np.where(PilotIndices, ReceivedSymbols, 0)
            TransmittedPilots = np.where(PilotIndices, metadata["Blocks"], 0)

            VarNoise = metadata["VarNoises"][snr_ind]
            VarNoiseShape = np.arange(self.LengthCP, self.LengthCP+self.NumSubCarrier)
            NoisePower = VarNoise[np.ix_(VarNoiseShape, VarNoiseShape)]

            EstimationMetadata = {
                "NoisePower": NoisePower,
                "mean_h": mean_h,
                "R_hh": R_hh
            }
            EstimatedChannel = self.ChannelEstimator.process(ReceivedPilots, TransmittedPilots, EstimationMetadata)
            EstimatedChannel = self.ChannelInterpolator.process(EstimatedChannel)
            EstimatedChannels.append(EstimatedChannel)
            # Equalization #
            RecoveredSymbols = self.Equalizer.process(EstimatedChannel, ReceivedSymbols)
            EstimatedDataSymbol = RecoveredSymbols[~PilotIndices].reshape(metadata["NumBlocks"],-1)
            CodedEstimatedBits = self.Demodulator.demodulate(EstimatedDataSymbol)
            CodedEstimatedBits = CodedEstimatedBits.flatten()[:-metadata["NumExtraBits"]]
            DeinterleavedEstimatedBits = self.Deinterleaver.deinterleave(CodedEstimatedBits, metadata)
            DecodedEstimatedBits = self.ChannelDecoder.decode(DeinterleavedEstimatedBits)
            EstimatedBits.append(DecodedEstimatedBits)
        
        EstimatedChannels = np.stack(EstimatedChannels, axis=0)
        EstimatedBits = np.stack(EstimatedBits, axis=0)
        return EstimatedBits, EstimatedChannels

class OFDMSystem():
    def __init__(self, config) -> None:
        self.SNRs = config.SNR
        self.Transmitter = Transmitter(config)
        self.ChannelBlock = ChannelBlock(config)
        self.Receiver = Receiver(config)
        self.Evaluater = TotalEvaluators(config.SNR)

    def run(self):
        OFDMSymbols, metadata = self.Transmitter.process()
        logging.info("Transmitted signals")
        ReceivedSignals, metadata = self.ChannelBlock.process(OFDMSymbols, metadata)
        logging.info("Received signals and start recovering signal")
        EstimatedBits, EstimatedChannels = self.Receiver.process(ReceivedSignals, metadata)
        logging.info("Data recovered")

        # Duplicate for each SNR
        DataBits = np.tile(metadata["DataBits"], (len(self.SNRs),1))
        ChannelFreqDomain = np.tile(metadata["ChannelsFreqDomain"], (len(self.SNRs),1,1))

        # Remove extra zero bits from transmitter
        EstimatedBits = EstimatedBits.reshape(len(self.SNRs), -1)[:,:metadata["DataBitSize"]]

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