import numpy as np
from enum import Enum
from abc import ABC, abstractmethod
from Modulators import *   

class PILOT_TYPES(Enum):
    BLOCK = "BlockTypePilot"
    COMB = "CombTypePilot"

class BitsGenerator():
    def __init__(self, NumBits: int) -> None:
        self.NumBits = NumBits

    def process(self) -> np.ndarray:
        return np.random.randint(0, 2, self.NumBits)

class BlockGenerators(ABC):
    def __init__(self, config):
        self.NumSubCarrier = config.NumSubCarrier
        self.NumPilotPerBlock = config.NumPilotPerBlock
        self.Modulator = SelectModulator(config.QAMModulation)

    @abstractmethod
    def InsertPilot(self, DataSymbols: np.ndarray, BitsPerSymbol: int) -> np.ndarray:
        ...

    @abstractmethod
    def ExtractData(self, ReceivedBlocks: np.ndarray, PilotIndices: np.ndarray, NumDataSymbols: int) -> np.ndarray:
        ...

class BlockTypePilot(BlockGenerators):
    
    def InsertPilot(self, DataSymbols: np.ndarray, BitsPerSymbol: int) -> tuple:
        NumDataBlocks = int(np.ceil(DataSymbols.size/self.NumSubCarrier))
        BlockShape = (1+NumDataBlocks, self.NumSubCarrier)
        Blocks = np.zeros(BlockShape, dtype=complex)

        # Pilot
        PilotIndices = np.zeros(BlockShape, dtype=bool)
        PilotIndices[0,:] = True

        NumPilotBits = self.NumSubCarrier * BitsPerSymbol
        PilotGenerator = BitsGenerator(NumPilotBits)
        PilotBits = PilotGenerator.process()
        Pilots = self.Modulator.modulate(PilotBits)
        Blocks[0,:] = Pilots

        # Data with zero padding to complete the last block
        NumExtraSymbols = NumDataBlocks*self.NumSubCarrier - DataSymbols.size
        NumExtraBits = NumExtraSymbols * BitsPerSymbol
        ExtraBits = np.zeros(NumExtraBits)
        ExtraSymbols = self.Modulator.modulate(ExtraBits)
        PaddedDataSymbols = np.concat((DataSymbols, ExtraSymbols))
        PaddedDataSymbols = PaddedDataSymbols.reshape(NumDataBlocks, self.NumSubCarrier)
        Blocks[1:,:] = PaddedDataSymbols
        return Blocks, PilotIndices
    
    def ExtractData(self, ReceivedBlocks: np.ndarray, PilotIndices: np.ndarray, NumDataSymbols: int) -> np.ndarray:
        EstimatedDataSymbols = ReceivedBlocks[~PilotIndices][:NumDataSymbols]
        return EstimatedDataSymbols
    
class CombTypePilot(BlockGenerators):
    def InsertPilot(self, DataSymbols: np.ndarray, BitsPerSymbol: int) -> tuple:
        NumDataPerBlock = self.NumSubCarrier - self.NumPilotPerBlock
        NumBlocks = int(np.ceil(DataSymbols.size/NumDataPerBlock))
        BlockShape = (NumBlocks, self.NumSubCarrier)
        Blocks = np.zeros(BlockShape, dtype=complex)

        # Pilot        
        PilotIndices = np.zeros(BlockShape, dtype=bool)
        PilotStep = self.NumSubCarrier//self.NumPilotPerBlock
        PilotPosition = np.arange(0, self.NumSubCarrier, PilotStep)[:self.NumPilotPerBlock]
        PilotIndices[:,PilotPosition] = True

        NumPilotBits = NumBlocks * self.NumPilotPerBlock * BitsPerSymbol
        PilotGenerator = BitsGenerator(NumPilotBits)
        PilotBits = PilotGenerator.process()
        Pilots = self.Modulator.modulate(PilotBits)
        Blocks[PilotIndices] = Pilots
        
        # Data with zero padding to complete the last block
        NumExtraSymbols = NumBlocks*NumDataPerBlock - DataSymbols.size
        NumExtraBits = NumExtraSymbols * BitsPerSymbol
        ExtraBits = np.zeros(NumExtraBits)
        ExtraSymbols = self.Modulator.modulate(ExtraBits)
        PaddedDataSymbols = np.concat((DataSymbols, ExtraSymbols))
        Blocks[~PilotIndices] = PaddedDataSymbols
        return Blocks, PilotIndices
    
    def ExtractData(self, ReceivedBlocks: np.ndarray, PilotIndices: np.ndarray, NumDataSymbols: int) -> np.ndarray:
        EstimatedDataSymbols = ReceivedBlocks[~PilotIndices][:NumDataSymbols]
        return EstimatedDataSymbols

def SelectPilotType(config) -> BlockGenerators:
    if config.PilotTypes == PILOT_TYPES.BLOCK: 
        return BlockTypePilot(config)
    elif config.PilotTypes == PILOT_TYPES.COMB: 
        return CombTypePilot(config)
    else: raise ValueError("Unsuport pilot type")

if __name__ == '__main__':

    class blckconfig:
        TotalBits = 2000
        TotalSymbols = 1000
        NumSubCarrier = 9
        TotalNumBlock = 6
        NumPilotPerBlock = 2
        PilotTypes = PILOT_TYPES.BLOCK
        QAMModulation = QAM_MODULATION.QPSK_GRAY

    class TestBlockGen():
        def __init__(self, blckconfig):
            self.BitGenerator = BitsGenerator(blckconfig.TotalBits)
            self.Modulator = SelectModulator(blckconfig.QAMModulation)
            self.PilotInsertion = SelectPilotType(blckconfig)
        
        def run(self):
            DataBits = self.BitGenerator.process()
            DataSymbols = self.Modulator.modulate(DataBits)
            Blocks, PilotIndices = self.PilotInsertion.InsertPilot(DataSymbols, self.Modulator.bitsPerSymbol)
            return Blocks

    system = TestBlockGen(blckconfig)
    result = system.run()
    print(result)