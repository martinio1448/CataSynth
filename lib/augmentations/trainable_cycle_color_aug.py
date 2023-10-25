
from typing import TypedDict
from typing_extensions import Unpack
from lib.augmentations.abstract_trainable_augmentation import AbstractTrainableAugmentation
from lib.augmentations.abstract_trainable_augmentation_set import AbstractTrainableAugmentationSet
from lib.augmentations.cycle_color import CycleColor
import torch

from lib.model.layers import ColorCycleEmbedding

class TrainableColorCycleArgs(TypedDict):
    cycle_pos: int
    cycle_length: int
    generation_range: int
    background_tolerance: int
    cycle_embedding_ch: int
    timestep_embedding_ch: int

class TrainableColorCycle(AbstractTrainableAugmentation):
    def __init__(self, **kwargs: Unpack[TrainableColorCycleArgs]) -> None:
        super().__init__()
        

        self.transform = CycleColor(epoch=kwargs['cycle_pos'], 
                                    background_tolerance=kwargs['background_tolerance'], 
                                    cycle=kwargs['cycle_length'], 
                                    generation_range=kwargs["generation_range"]
                                    )
        
        self.embedding = torch.nn.ModuleList([
            ColorCycleEmbedding(kwargs['cycle_embedding_ch']),
            torch.nn.Linear(kwargs['cycle_embedding_ch'],
                            kwargs['timestep_embedding_ch']),
            torch.nn.Linear(kwargs['timestep_embedding_ch'],
                            kwargs['timestep_embedding_ch']),
        ])