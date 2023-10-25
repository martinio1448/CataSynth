
from lib.augmentations.abstract_trainable_augmentation_set import AbstractTrainableAugmentationSet

class AbstractTrainableAugmentation:
    def __init__(self) -> None:

    def init_transform(self):
        self.transform = None

    def init_handler(self):
        self.handler = None