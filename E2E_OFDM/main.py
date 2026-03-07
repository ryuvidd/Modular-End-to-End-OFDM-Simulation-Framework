from dataclasses import dataclass
from Modulators import *
from Channels import *
from ChannelEstimators import *
from EvaluationMetric import *
from BlocksGenerator import *
from Equalizers import *

if __name__ == '__main__':
    
    @dataclass
    class OFDMConfig:
        BlockConfiguration: BlockConfig = BlockConfig(
            NumSubCarrier = 64, 
            TotalNumBlock = 10000,
            NumPilotPerBlock = 16,
            PilotTypes = PILOT_TYPES.COMB
        )
        QAMModulation: QAM_MODULATION = QAM_MODULATION.QPSK_GRAY
        LengthCP: int = 10   # must satisfy L_CP >= L_h - 1 where L_h - 1 is max delay spread, commonly L_CP = N/4, N/8, N/16
        ChannelConfiguration: ChannelConfig = ChannelConfig(
            Model = CHANNEL_MODEL.RAYLEIGH,
            NumTap = 4,
            RegenChannel = 5
        )
        SNR: int = 10
        Estimator: CHANNEL_ESTIMATOR = CHANNEL_ESTIMATOR.LS

    class OFDMSystem():
        def __init__(self, config) -> None:
            self.BlocksGenerator = SelectPilotType(config.BlockConfiguration)
            self.Modulator = SelectModulator(config.QAMModulation)
            self.OFDMModulator = OFDMModulator(config.LengthCP)
            self.ChannelGenerator = SelectChannelModel(config.ChannelConfiguration)
            if config.ChannelConfiguration.Model == CHANNEL_MODEL.NODISTORTION: self.Noise = AWGNChannel(1000)
            else: self.Noise = AWGNChannel(config.SNR)
            self.ChannelEstimator = SelectEstimator(config.Estimator)
            self.Equalizer = ZeroForcing()
            self.Evaluater = TotalEvaluators()            

        def run(self):
            # Transmitter side #
            Blocks, DataBits = self.BlocksGenerator.process(self.Modulator.bitsPerSymbol)
            TransmittedSymbols = self.Modulator.modulate(Blocks)
            OFDMSymbols = self.OFDMModulator.modulate(TransmittedSymbols)
            # Channel #
            ChannelOutput = self.ChannelGenerator.process(OFDMSymbols, TransmittedSymbols.shape[1])
            ReceivedSignal= self.Noise.process(ChannelOutput, ChannelOutput.shape[1])
            # Receiver side #
            ReceivedSymbols = self.OFDMModulator.demodulate(ReceivedSignal)
            # Channel Estimation #
            ReceivedPilots = np.where(self.BlocksGenerator.PilotIndices, ReceivedSymbols, 0)
            TransmittedPilots = np.where(self.BlocksGenerator.PilotIndices, TransmittedSymbols, 0)
            EstimatedChannel = self.ChannelEstimator.process(ReceivedPilots, TransmittedPilots)
            # Equalization #
            RecoveredSymbols = self.Equalizer.process(EstimatedChannel, ReceivedSymbols)
            EstimatedDataSymbol = RecoveredSymbols[~self.BlocksGenerator.PilotIndices].reshape(Blocks.shape[0],-1)
            EstimatedBits = self.Modulator.demodulate(EstimatedDataSymbol)

            EstimatedData = ExperimentData(
                Bits=EstimatedBits, 
                Channel=EstimatedChannel
            )
            GroundTruthData = ExperimentData(
                Bits=DataBits, 
                Channel=np.repeat(np.fft.fft(self.ChannelGenerator.Channels), self.ChannelGenerator.RegenChannel, axis=0)
            )
            results = self.Evaluater.process(EstimatedData, GroundTruthData)
            return results

    system = OFDMSystem(OFDMConfig())
    Result = system.run()
    print("BER: {} \nNMSE: {} dB".format(Result.BER, Result.ChannelNMSE))