from dataclasses import dataclass
from Modulators import *
from Channels import *
from ChannelEstimators import *
from ChannelInterpolator import *
from EvaluationMetric import *
from BlocksGenerator import *
from Equalizers import *
from util import *

if __name__ == '__main__':

    @dataclass
    class OFDMconfig:
        NumSubCarrier: int = 64
        TotalNumBlock: int = 10000
        NumPilotPerBlock: int = 16
        QAMModulation: QAM_MODULATION = QAM_MODULATION.QPSK_GRAY
        LengthCP: int = 4
        SNR: np.ndarray = np.arange(-10,11,2)
        BlockConfiguration: BlockConfig = BlockConfig(
            NumSubCarrier = NumSubCarrier, 
            TotalNumBlock = TotalNumBlock,
            NumPilotPerBlock = NumPilotPerBlock,
            PilotTypes = PILOT_TYPES.COMB
        )
        ChannelBlockConfig = ChannelConfig(
            Model = CHANNEL_MODEL.RAYLEIGH,
            NumTap = 3,
            RegenChannel = 5
        )
        Estimator: CHANNEL_ESTIMATOR = CHANNEL_ESTIMATOR.LS
        Interpolator: CHANNEL_INTERPOLATOR = CHANNEL_INTERPOLATOR.LINEAR

    class Transmitter():
        def __init__(self, config) -> None:
            self.BlocksGenerator = SelectPilotType(config.BlockConfiguration)
            self.Modulator = SelectModulator(config.QAMModulation)
            self.OFDMModulator = OFDMModulator(config.LengthCP, config.NumSubCarrier)

        def process(self, OFDMconfig) -> tuple:
            Blocks, DataBits = self.BlocksGenerator.process(self.Modulator.bitsPerSymbol)
            TransmittedSymbols = self.Modulator.modulate(Blocks)
            OFDMSymbols = self.OFDMModulator.modulate(TransmittedSymbols)
            OFDMconfig.PilotIndices = self.BlocksGenerator.PilotIndices
            OFDMconfig.TransmittedSymbols = TransmittedSymbols
            return OFDMSymbols, DataBits, OFDMconfig
        
    class ChannelBlock():
        def __init__(self, ChannelBlockConfig) -> None:
            self.ChannelGenerator = SelectChannelModel(ChannelBlockConfig)
            self.NoiseMixer = NoiseMixer()

        def process(self, OFDMSymbols: np.ndarray, OFDMconfig) -> tuple:
            ChannelOutput = self.ChannelGenerator.process(OFDMSymbols, OFDMconfig.NumSubCarrier)
            ChannelsFreq = self.ChannelGenerator.Channels_feq
            ReceivedSignals = []
            for snr in OFDMconfig.SNR:
                ReceivedSignals.append(self.NoiseMixer.process(ChannelOutput, snr))
            ReceivedSignals = np.stack(ReceivedSignals, axis=0)
            return ReceivedSignals, ChannelsFreq
        
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
                EstimatedChannel = self.ChannelEstimator.process(ReceivedPilots, TransmittedPilots)
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
            self.ChannelBlock = ChannelBlock(config.ChannelBlockConfig)
            self.Receiver = Receiver(config)
            self.Evaluater = TotalEvaluators(config.SNR)
            self.Plotter = Plotter()

        def run(self):
            logging.info("Initialize OFDM system")
            OFDMSymbols, DataBits, config = self.Transmitter.process(self.OFDMconfig)
            logging.info("Transmitted signals")
            ReceivedSignals, ChannelFreqDomain = self.ChannelBlock.process(OFDMSymbols, config)
            logging.info("Received signals and start recovering signal")
            EstimatedBits, EstimatedChannels = self.Receiver.process(ReceivedSignals, config)

            # Duplicate for each SNR
            DataBits = np.tile(DataBits, (len(self.OFDMconfig.SNR),1,1))
            ChannelFreqDomain = np.tile(ChannelFreqDomain, (len(self.OFDMconfig.SNR),1,1))

            EstimatedData = ExperimentData(
                Bits=EstimatedBits, 
                Channel=EstimatedChannels
            )
            GroundTruthData = ExperimentData(
                Bits=DataBits, 
                Channel=ChannelFreqDomain
            )
            results = self.Evaluater.process(EstimatedData, GroundTruthData)

            logging.info("Visualizing results")
            self.Plotter.plot_BER(self.OFDMconfig.SNR, results.BER, save_fig_name='results/BER.png')            
            self.Plotter.plot_NMSE(self.OFDMconfig.SNR, results.ChannelNMSE, save_fig_name='results/NMSE.png')            

    system = OFDMSystem(OFDMconfig)
    system.run()