from dataclasses import dataclass, field

import torch
from nestconfig import NestConfig
from typing import Union


@dataclass
class DataConfig:
    DATA_ROOT_PATH = "/home/s4076893/Desktop"  # path where all datasets are saved
    DATASET_CLASS: str = "EpicKitchens"
    FEAT_DIR = "epickitchens100/features/rgb_kinetics_bninception"
    DROP_LAST: bool = False

    # Short-term specific
    TAU_A: float = 1.  # ahead of true action, in seconds
    TAU_O: float = 69.  # length of observation, in seconds (working mem + long mem)
    PAST_STEP_IN_SEC: float = 0.25
    PAST_SAMPLE_RATE: int = 1
    FUTURE_STEP_IN_SEC: float = 1.
    FUTURE_SAMPLE_RATE: int = 4  # step_in_sec * feature_fps
    LONG_MEMORY_LENGTH: float = 64.  # secs of long-term memory, if fps is 4 -> 256 sequence length

    # For thumos or online action detection
    RGB_TYPE: str = 'rgb_kinetics_resnet50'
    FLOW_TYPE: str = 'flow_kinetics_bninception'
    STRIDE: float = 1.  # stride in seconds
    DATA_NAME: str = 'THUMOS'
    NUM_CLASSES: int = 22  # including background


@dataclass
class DiverseSetConfig:
    # ENABLE turns on diverse-set evaluation: inference-time greedy rerank,
    # set-recall / duplicate / object-match metrics. It does NOT change the
    # decoder architecture and works with a single-query checkpoint.
    ENABLE: bool = False
    # MULTI_QUERY swaps the decoder to SET_SIZE parallel future queries so
    # pred.future_actions has shape [B, SET_SIZE, num_actions]. It changes
    # weight shapes -- training a checkpoint at MULTI_QUERY=False and loading
    # it at MULTI_QUERY=True relies on helper.load_model's shape filter.
    MULTI_QUERY: bool = False
    SET_SIZE: int = 5
    # Weight of the Step-8 fixed-role coverage loss (> 0 switches the future head
    # from hit-anywhere WTA to fixed-role: slot 0 is anchored to the GT with
    # equalized CE, slots 1..K-1 are matched to slot 0's detached top-(K-1)
    # non-GT modes). 0 disables it and reverts to the Step-7 WTA ablation.
    # Suggested starting range 0.5-1.0 (it is a per-slot CE, comparable in scale
    # to the anchor CE).
    DIVERSITY_WEIGHT: float = 0.
    # Temperature of slot 0's detached distribution softmax(z_0 / tau) used to
    # RANK its runner-up modes for assignment. tau < 1 sharpens the ranking
    # toward slot 0's most confident modes; tau = 1.0 uses the raw distribution.
    # It does not sharpen the trained (slots 1..K-1) distributions.
    DIVERSITY_TEMP: float = 1.0
    # Linear warmup (in epochs) for the coverage weight: it ramps 0 -> 1 over the
    # first N epochs so slot 0 becomes a meaningful ranker before slots 1..K-1
    # are asked to mirror its modes. 0 disables warmup (full weight from epoch 0).
    DIVERSITY_WARMUP_EPOCHS: int = 5
    HIT_WEIGHT: float = 0.
    # Epsilon-relaxed WTA for MultipSetHitLoss:
    #   loss = (1 - eps) * min_k CE_k + eps * mean_k CE_k
    # eps = 0.0 = pure "hit-anywhere" (only the winning slot is supervised).
    # Raise to 0.05-0.2 if you observe slot starvation during training.
    HIT_EPSILON: float = 0.0
    OBJECT_WEIGHT: float = 0.
    TEMPORAL_WEIGHT: float = 0.

    # First-order action-transition prior P(next action | previous action),
    # built once from EK100 training sequences. Because it is external/data-driven
    # (not slot-0 self-distillation), it lets the diverse set cover true conditional
    # modes the single head ranks below its top-K: transition recall@5 ~34.9 vs the
    # head's topk_set_recall@5 ~22.4 on validation.
    #
    # Step A (inference probe): fuse into eval logits, score = head_logit +
    # TRANSITION_WEIGHT * log P(next|prev). 0 disables (eval unchanged).
    TRANSITION_WEIGHT: float = 0.
    # Which previous action indexes the prior at eval: "gt_prev" = last observed GT
    # action (legitimate in anticipation; upper bound on realizable gain);
    # "past_head" = the model's own past-head distribution (end-to-end, soft mixture).
    TRANSITION_FUSE_SOURCE: str = "gt_prev"
    # Step B (training): source of the fixed-role coverage modes. "self" = slot 0's
    # detached top-(K-1) modes (Step 8(5), capped at the single-head top-K);
    # "transition" = top-(K-1) transition successors of the observed action (external
    # targets that can push the diverse set above the single-head ceiling).
    COVERAGE_SOURCE: str = "self"
    # Directory holding training.csv for the transition prior (mirrors the equalized
    # loss default). Built lazily only when TRANSITION_WEIGHT>0 or COVERAGE_SOURCE=="transition".
    TRANSITION_ANNO: str = "annotations/ek100_rulstm/"


@dataclass
class ModelConfig:
    ENCODER_CLASS: str = "MAMBA"
    MAMBA_VERSION: int = 1
    CRITERION_CLASS: str = "Criterion"
    INPUT_DIM: int = 1024
    D_MODEL: int = 512
    N_LAYER: int = 2
    N_DEC_LAYER: int = 2
    IGNORE_INDEX: int = -1  # class that does not contribute to the loss
    D_FFN: int = 2048
    N_HEADS: int = 8

    SHARE_CLASSIFIER: bool = False
    PAST_CLS: bool = True

    # Action, Verb, Noun Classification
    ACTION_CLS: bool = True
    VERB_CLS: bool = False
    NOUN_CLS: bool = False

    # Past classification
    CLS_WORK: bool = True
    CLS_LAST: bool = False
    CLS_ALL: bool = False

    DROPOUT: float = 0.
    DROP_CLS: float = 0.
    DROP_DEC: float = 0.1

    # MAMBA
    D_STATE: int = 64
    D_CONV: int = 4
    RETURN_INTERMEDIATE: bool = False
    INTERMEDIATE_LAYER_IDX: int = -1

    # For querydecoder
    PRENORM: bool = False
    N_QUERIES: int = 1

    # Diverse next-action set prediction scaffolding
    DIVERSE_SET: DiverseSetConfig = field(default_factory=DiverseSetConfig)

    # activation
    ACTIVATION: str = 'relu'

    # Uniformly sample from the past instead of last tokens
    SAMPLE_UNIFORM: bool = False


@dataclass
class ClusteringConfig:
    ENABLE: bool = False
    ON_FRAME_TOKENS: bool = True
    N_CLUSTERS: int = 10
    USE_EMBEDDING: bool = True
    CAT_WORK_LAST: bool = False
    GATE_STATE: bool = True
    EXPAND_K: float = 0.5
    USE_SCAN: bool = False
    LAYERS: int = 1


@dataclass
class TrainConfig:
    ENABLE: bool = True
    CKPT_PATH: str = None
    BATCH_SIZE: int = 64
    NUM_WORKERS: int = 8
    OPTIMIZER: str = "sgd"
    WEIGHT_DECAY: float = 0.
    SCHEDULER: str = "cosine"
    EPOCHS: int = 50
    WARMUP_STEPS: int = 5
    LR: float = 0.001
    MIN_LR: float = 1e-7
    GRADIENT_CLIPPING: Union[float, None] = None
    USE_MIXUP: bool = False
    SAVE_MODEL: bool = True


@dataclass
class ValConfig:
    ENABLE: bool = True
    BATCH_SIZE: int = 64
    NUM_WORKERS: int = 8
    EVALUATE_EVERY: int = 1
    MAX_LEN: int = 1


@dataclass
class TestConfig:
    ENABLE: bool = False
    CKPT_PATH: str = None


@dataclass
class Config(NestConfig):
    SEED = 42
    PRIMARY_METRIC = "val/mt5r"
    NOTE = None   # some notes of the experiment
    USE_WANDB = True  # whether to use wandb to visualize logs
    LOG_LEVEL = 'info'  # info or debug
    WANDB_PROJECT = None
    METRIC_DESCENDING: bool = False
    DTYPE: str = "float32"

    MODEL: ModelConfig = field(default_factory=ModelConfig)
    TRAIN: TrainConfig = field(default_factory=TrainConfig)
    VAL: ValConfig = field(default_factory=ValConfig)
    TEST: TestConfig = field(default_factory=TestConfig)
    DATA: DataConfig = field(default_factory=DataConfig)
    CLUSTERING: ClusteringConfig = field(default_factory=ClusteringConfig)
