"""
    Register PIXEL models for the AutoClass interface.
    https://huggingface.co/docs/transformers/en/model_doc/auto
"""

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForPreTraining,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)
from mpixel.models.mpixel.configuration_mpixel import MPIXELConfig
from mpixel.models.mpixel.modeling_mpixel import (
    MPIXELForPreTraining,
    MPIXELForSequenceClassification,
    MPIXELForTokenClassification,
    MPIXELModel,
)


AutoConfig.register("mpixel", MPIXELConfig)
AutoModel.register(MPIXELConfig, MPIXELModel)
AutoModelForPreTraining.register(MPIXELConfig, MPIXELForPreTraining)
AutoModelForSequenceClassification.register(MPIXELConfig, MPIXELForSequenceClassification)
AutoModelForTokenClassification.register(MPIXELConfig, MPIXELForTokenClassification)
