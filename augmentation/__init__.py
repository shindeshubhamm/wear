"""
Augmentation Module
===================
"""


augmentation_choices = ['autoaughar', 'cutmix', 'jittering', 'mixup']
"""
List of valid augmentation choices

Valid choices for augmentation techniques:
- 'autoaughar': Use AutoAugHAR for augmentation.
- 'cutmix': Perform CutMix augmentation.
- 'jittering': Apply jittering augmentation.
- 'mixup': Perform MixUp augmentation.
"""

from .augment import apply_augmentation


__all__ = ['augmentation_choices', 'apply_augmentation']