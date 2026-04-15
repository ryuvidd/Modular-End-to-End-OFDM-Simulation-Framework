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
        NumPilotPerBlock: int = 32
        QAMModulation: QAM_MODULATION = QAM_MODULATION.QPSK_GRAY
        LengthCP: int = 8
        SNR: np.ndarray = np.arange(-10,11,2)
        BlockConfiguration: BlockConfig = BlockConfig(
            NumSubCarrier = NumSubCarrier, 
            TotalNumBlock = TotalNumBlock,
            NumPilotPerBlock = NumPilotPerBlock,
            PilotTypes = PILOT_TYPES.COMB
        )
        ChannelBlockConfig = ChannelConfig(
            Model = CHANNEL_MODEL.RAYLEIGH,
            NumTap = 5,
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
            self.ChannelBlock = ChannelBlock(config.ChannelBlockConfig)
            self.Receiver = Receiver(config)
            self.Evaluater = TotalEvaluators(config.SNR)
            self.Plotter = Plotter()

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

    totalconfig = [
        OFDMconfig(Estimator=CHANNEL_ESTIMATOR.LS),
        OFDMconfig(Estimator=CHANNEL_ESTIMATOR.LMMSE)
    ]

    BERs = []
    NMSEs = []
    for k in range(len(totalconfig)):
        logging.info(f"===== Initialize OFDM system {k+1} =====")
        system = OFDMSystem(totalconfig[k])
        result = system.run()
        logging.info(f"===== Successfully simulate OFDM system {k+1} =====\n")
        BERs.append(result.BER)
        NMSEs.append(result.ChannelNMSE)

    MergedResults = {
        "BERs": np.stack(BERs, axis=0),
        "ChannelNMSEs": np.stack(NMSEs, axis=0)
    }

    logging.info("Visualizing results\n")
    CurvesPlotter = Plotter()
    labels = ["LS", "LMMSE"]
    CurvesPlotter.plot_BER(
        SNR=OFDMconfig.SNR, 
        BER=MergedResults["BERs"], 
        label=labels,
        save_fig_name='results/BER_.png'
    )
    CurvesPlotter.plot_NMSE(
        SNR=OFDMconfig.SNR, 
        NMSE=MergedResults["ChannelNMSEs"], 
        label=labels,
        save_fig_name='results/NMSE_.png'
    )            