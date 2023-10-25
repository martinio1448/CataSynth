from collections.abc import Iterable


class AbstractTrainableAugmentationSet():
    def __init__(self, augmentations: Iterable[AbstractTrainableAugmentation]) -> None:
        self.augmentations = augmentations