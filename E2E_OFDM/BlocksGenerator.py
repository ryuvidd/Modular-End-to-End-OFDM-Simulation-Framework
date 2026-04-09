import numpy as np
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod
from Modulators import *   

class PILOT_TYPES(Enum):
    BLOCK = "BlockTypePilot"
    COMB = "CombTypePilot"

@dataclass
class BlockConfig:
    NumSubCarrier: int 
    TotalNumBlock: int
    NumPilotPerBlock: int
    PilotTypes: PILOT_TYPES

# Here says BitsGenerator is a 'block', therefore it enforces the only method called 'process'.    
class BitsGenerator():
    def process(self, NumBlocks: int, NumBit: int) -> np.ndarray:
        return np.random.randint(0, 2, [NumBlocks, NumBit])

class BlockGenerators(ABC):
    def __init__(self, config: BlockConfig):
        self.TotalNumBlock = config.TotalNumBlock
        self.NumSubCarrier = config.NumSubCarrier
        self.NumPilotPerBlock = config.NumPilotPerBlock
        self.BitGenerator = BitsGenerator()
        self.PilotIndices = np.zeros((config.TotalNumBlock, config.NumSubCarrier), dtype=bool)

    @abstractmethod
    def process(self, BitsPerSymbol: int) -> tuple:
        ...

class BlockTypePilot(BlockGenerators):
    def process(self, BitsPerSymbol: int) -> tuple:
        self.PilotIndices[0,:] = 1
        Blocks = np.zeros([self.TotalNumBlock, self.NumSubCarrier * BitsPerSymbol], dtype=int)
        # Pilot
        Blocks[0,:] = self.BitGenerator.process(1, self.NumSubCarrier * BitsPerSymbol)
        # Data
        Data = self.BitGenerator.process(self.TotalNumBlock-1, self.NumSubCarrier * BitsPerSymbol)
        Blocks[1:,:] = Data
        return Blocks, Data
    
class CombTypePilot(BlockGenerators):
    def process(self, BitsPerSymbol: int) -> tuple:
        self.PilotIndices[:,::self.NumSubCarrier//self.NumPilotPerBlock] = 1
        mask = np.repeat(self.PilotIndices, BitsPerSymbol, axis=1)
        Blocks = np.zeros([self.TotalNumBlock, self.NumSubCarrier * BitsPerSymbol], dtype=int)
        # Pilot
        Pilots = self.BitGenerator.process(self.TotalNumBlock, self.NumPilotPerBlock * BitsPerSymbol)
        Blocks[mask] = Pilots.reshape(-1)
        # Data
        Data = self.BitGenerator.process(self.TotalNumBlock, (self.NumSubCarrier-self.NumPilotPerBlock) * BitsPerSymbol)
        Blocks[~mask] = Data.reshape(-1)
        return Blocks, Data

def SelectPilotType(config: BlockConfig) -> BlockGenerators:
    if config.PilotTypes == PILOT_TYPES.BLOCK: 
        return BlockTypePilot(config)
    elif config.PilotTypes == PILOT_TYPES.COMB: 
        return CombTypePilot(config)
    else: raise ValueError("Unsuport pilot type")

if __name__ == '__main__':

    blckconfig = BlockConfig(
        NumSubCarrier = 8,
        TotalNumBlock = 6,
        NumPilotPerBlock = 2,
        PilotTypes = PILOT_TYPES.BLOCK
    )

    class TestBlockGen():
        def __init__(self, blckconfig, QAMMapper):
            self.Modulator = SelectModulator(QAMMapper)
            self.BlockGenerator = SelectPilotType(blckconfig)
        
        def run(self):
            Blocks = self.BlockGenerator.process(self.Modulator.bitsPerSymbol)
            return Blocks

    system = TestBlockGen(blckconfig, QAM_MODULATION.QPSK_GRAY)
    result = system.run()
    print(result)