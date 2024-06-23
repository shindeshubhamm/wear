"""
Augmentation Module
===================
"""


augmentation_choices = ['autoaughar', 'cutmix', 'jittering', 'mixup', 's1', 's2']
"""
List of valid augmentation choices

Valid choices for augmentation techniques:
- 'cutmix': Perform CutMix augmentation.
- 'jittering': Apply jittering augmentation.
- 'mixup': Perform MixUp augmentation.
- 's1': Augment arm modality only using cutmix.
- 's2': Augment leg modality only using cutmix.
"""

from .augment import apply_augmentation


__all__ = ['augmentation_choices', 'apply_augmentation']