import os

os.environ["USE_JAX"] = "1"
os.environ["USE_TORCH"] = "1"
os.environ["USE_TF"] = "0"

try:
    import jax.tools.colab_tpu

    jax.tools.colab_tpu.setup_tpu()
    print(jax.devices())
except:
    print("Jax can't load the colab tpu!")
# !/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Pretraining the library models for T5-like span-masked language modeling on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be pretrained by this script:
https://huggingface.co/models?filter=t5
"""
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field

# You can also adapt this script on your own masked language modeling task. Pointers for this are left as comments.
from enum import Enum
from itertools import chain
from pathlib import Path
from typing import Dict, List, Any, NamedTuple, Optional, Tuple

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

import flax
import jax
import jax.numpy as jnp
import optax
from flax import jax_utils, traverse_util
from flax.training import train_state
from flax.training.common_utils import get_metrics, onehot, shard
from huggingface_hub import Repository
from transformers import (
    CONFIG_MAPPING,
    FLAX_MODEL_FOR_MASKED_LM_MAPPING,
    AutoTokenizer,
    BatchEncoding,
    FlaxT5ForConditionalGeneration,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    T5Config,
    is_tensorboard_available,
    set_seed,
    TensorType,
)
from transformers.file_utils import get_full_repo_name
from transformers.models.t5.modeling_flax_t5 import shift_tokens_right

from optimizers.optax import distributed_shampoo
from jax import lax
import math
from torch.utils.data import DataLoader
from torch import manual_seed
from torch import (
    Generator as TorchGenerator,
    randint as torchRandInt,
    randperm as torchRandPerm,
)
from torch.utils.data import Sampler as TorchSampler
from torch import int64 as torchInt64
from flax.serialization import to_bytes, from_bytes
from multiprocessing.pool import ThreadPool as Pool
import datasets as hfds
import transformers as hftf
from t5_partitions import set_partitions
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze


MODEL_CONFIG_CLASSES = list(FLAX_MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class TrainingArguments:
    output_dir: str = field(
        default=None,
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written."
        },
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory. "
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )
    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_eval: bool = field(
        default=False, metadata={"help": "Whether to run eval on the dev set."}
    )
    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )

    learning_rate: float = field(
        default=5e-5, metadata={"help": "The initial learning rate for AdamW."}
    )
    learning_schedule: str = field(
        default="linear_warmup_decay",
        metadata={"help": "Which leanring rate schedule to use during training."},
    )
    warmup_steps: int = field(
        default=0, metadata={"help": "Linear warmup over warmup_steps."}
    )

    optim: str = field(
        default="adafactor",
        metadata={"help": "Which optimizer to use during training."},
    )
    optim_weight_decay: float = field(
        default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."}
    )
    optim_beta1: float = field(
        default=0.9, metadata={"help": "Beta1 for AdamW and SM3 optimizer."}
    )
    optim_beta2: float = field(
        default=0.999, metadata={"help": "Beta2 for AdamW and SM3 optimizer"}
    )
    optim_epsilon: float = field(
        default=1e-8, metadata={"help": "Epsilon for AdamW, shampoo and SM3 optimizer."}
    )

    shampoo_block_size: int = field(
        default=128,
        metadata={
            "help": "Chunked size for large layers with Distributed Shampoo. Use 128 as default (increase if you have compute budget)."
        },
    )
    shampoo_start_preconditioning_step: int = field(
        default=100,
        metadata={"help": "Number of steps before starting to update preconditioner."},
    )
    shampoo_preconditioning_compute_steps: int = field(
        default=10, metadata={"help": "Number of steps to update preconditioner."}
    )
    shampoo_skip_preconditioning_dim_size_gt: int = field(
        default=4096,
        metadata={"help": "Max size for preconditioning with Distributed Shampoo."},
    )
    shampoo_optim_quantized: bool = field(
        default=False,
        metadata={
            "help": "Whether to quantize optimizer (only supported with Distributed Shampoo)."
        },
    )
    shampoo_moving_average_for_momentum: bool = field(
        default=False,
        metadata={
            "help": "Whether to use moving average for momentum instead of exponential moving average."
        },
    )
    shampoo_nesterov: bool = field(
        default=False,
        metadata={"help": "Nesterov momentum."},
    )

    num_train_epochs: float = field(
        default=3.0, metadata={"help": "Total number of training epochs to perform."}
    )
    logging_steps: int = field(
        default=500, metadata={"help": "Log every X updates steps."}
    )
    save_steps: int = field(
        default=500, metadata={"help": "Save checkpoint every X updates steps."}
    )
    eval_steps: int = field(
        default=None, metadata={"help": "Run an evaluation every X steps."}
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed that will be set at the beginning of training."},
    )
    push_to_hub: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to upload the trained model to the model hub after training."
        },
    )
    hub_model_id: str = field(
        default=None,
        metadata={
            "help": "The name of the repository to keep in sync with the local `output_dir`."
        },
    )
    hub_token: str = field(
        default=None, metadata={"help": "The token to use to push to the Model Hub."}
    )
    private_repo: bool = field(
        default=False, metadata={"help": "Whether it is a private repo or not."}
    )
    resume_from_checkpoint: bool = field(
        default=False,
        metadata={"help": "Whether to resume the model from the cloned repo or not."},
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={
            "help": "Number of forward pass to perform before running backprop. This is useful to emulate large batch size."
        },
    )
    skip_memory_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to monitor jax TPU usage or not."},
    )

    def __post_init__(self):
        if self.output_dir is not None:
            self.output_dir = os.path.expanduser(self.output_dir)

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
        return d


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: "
            + ", ".join(MODEL_TYPES)
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": "Floating-point format in which the model weights should be initialized and trained. Choose one of `[float32, float16, bfloat16]`."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."
        },
    )
    train_ref_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input train ref data file for whole word masking in Chinese."
        },
    )
    validation_ref_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input validation ref data file for whole word masking in Chinese."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization and masking. Sequences longer than this will be truncated. Default to the max input length of the model."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    masking_strategy: str = field(
        default="span",
        metadata={
            "help": "Which masking method will be use during training (span,  n-gram-tokens, 1-gram-amino or 1-gram-amino-eq, n-gram-tokens-eq)."
        },
    )
    mlm_probability: float = field(
        default=0.15,
        metadata={
            "help": "Ratio of tokens to mask for span masked language modeling loss"
        },
    )
    mean_noise_span_length: float = field(
        default=3.0,
        metadata={"help": "Mean span length of masked tokens"},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    line_by_line: bool = field(
        default=False,
        metadata={
            "help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."
        },
    )
    reconstruct_whole_input: bool = field(
        default=False,
        metadata={
            "help": "Whether to reconstruct the whole input for the 1/n-gram method or mask the non-noised input in the output using cross entropy ignore index (-100) to ignore them in the loss."
        },
    )
    data_loader_prefetch_factor: int = field(
        default=2,
        metadata={"help": "Number of samples loaded in advance by each worker."},
    )
    data_loader_num_workers: int = field(
        default=8,
        metadata={
            "help": "how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        },
    )

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
        ):
            raise ValueError(
                "Need either a dataset name or a training/validation file."
            )
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                    "txt",
                ], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                    "txt",
                ], "`validation_file` should be a csv, a json or a txt file."


def compute_input_and_target_lengths(
    inputs_length, noise_density, mean_noise_span_length
):
    """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2466>`__ .

    Training parameters to avoid padding with random_spans_noise_mask.
    When training a model with random_spans_noise_mask, we would like to set the other
    training hyperparmeters in a way that avoids padding.
    This function helps us compute these hyperparameters.
    We assume that each noise span in the input is replaced by extra_tokens_per_span_inputs sentinel tokens,
    and each non-noise span in the targets is replaced by extra_tokens_per_span_targets sentinel tokens.
    This function tells us the required number of tokens in the raw example (for split_tokens())
    as well as the length of the encoded targets. Note that this function assumes
    the inputs and targets will have EOS appended and includes that in the reported length.

    Args:
        inputs_length: an integer - desired length of the tokenized inputs sequence
        noise_density: a float
        mean_noise_span_length: a float
    Returns:
        tokens_length: length of original text in tokens
        targets_length: an integer - length in tokens of encoded targets sequence
    """

    def _tokens_length_to_inputs_length_targets_length(tokens_length):
        num_noise_tokens = int(round(tokens_length * noise_density))
        num_nonnoise_tokens = tokens_length - num_noise_tokens
        num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))
        # inputs contain all nonnoise tokens, sentinels for all noise spans
        # and one EOS token.
        _input_length = num_nonnoise_tokens + num_noise_spans + 1
        _output_length = num_noise_tokens + num_noise_spans + 1
        return _input_length, _output_length

    tokens_length = inputs_length

    while (
        _tokens_length_to_inputs_length_targets_length(tokens_length + 1)[0]
        <= inputs_length
    ):
        tokens_length += 1

    inputs_length, targets_length = _tokens_length_to_inputs_length_targets_length(
        tokens_length
    )

    # minor hack to get the targets length to be equal to inputs length
    # which is more likely to have been set to a nice round number.
    if noise_density == 0.5 and targets_length > inputs_length:
        tokens_length -= 1
        targets_length -= 1
    return tokens_length, targets_length


@flax.struct.dataclass
class FlaxDataCollatorForT5MLM:
    """
    Data collator used for T5 span-masked language modeling.
    It is made sure that after masking the inputs are of length `data_args.max_seq_length` and targets are also of fixed length.
    For more information on how T5 span-masked language modeling works, one can take a look
    at the `official paper <https://arxiv.org/pdf/1910.10683.pdf>`__
    or the `official code for preprocessing <https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py>`__ .

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        noise_density (:obj:`float`):
            The probability with which to (randomly) mask tokens in the input.
        mean_noise_span_length (:obj:`float`):
            The average span length of the masked tokens.
        input_length (:obj:`int`):
            The expected input length after masking.
        target_length (:obj:`int`):
            The expected target length after masking.
        pad_token_id: (:obj:`int`):
            The pad token id of the model
        decoder_start_token_id: (:obj:`int):
            The decoder start token id of the model
        masking_type: (:obj:`str):
            Type of masking method "span,  n-gram-tokens, 1-gram-amino or 1-gram-amino-eq, n-gram-tokens-eq"
        line_by_line: (:obj:`str):
            If we concat samples or not
    """

    tokenizer: PreTrainedTokenizerBase
    noise_density: float
    mean_noise_span_length: float
    input_length: int
    target_length: int
    pad_token_id: int
    decoder_start_token_id: int
    masking_type: str
    line_by_line: str
    max_length: int
    reconstruct_whole_input: bool = False
    pad_to_multiple_of: int = 8

    def __call__(self, examples: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:

        if self.masking_type in [
            "n-gram-tokens",
            "1-gram-amino",
            "1-gram-amino-eq",
            "n-gram-tokens-eq",
        ]:
            # Handle dict or lists with proper padding and conversion to tensor.
            batch = self.tokenizer.pad(
                examples,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=TensorType.NUMPY,
            )

            # If special token mask has been preprocessed, pop it from the dict.
            special_tokens_mask = batch.pop("special_tokens_mask", None)

            batch["input_ids"], batch["labels"] = self.mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        elif self.masking_type == "span":
            # convert list to dict and tensorize input
            batch = BatchEncoding(
                {
                    k: np.array([examples[i][k] for i in range(len(examples))])
                    for k, v in examples[0].items()
                }
            )

            input_ids = batch["input_ids"]
            batch_size, expandend_input_length = input_ids.shape

            mask_indices = np.asarray(
                [
                    self.random_spans_noise_mask(expandend_input_length)
                    for i in range(batch_size)
                ]
            )
            labels_mask = ~mask_indices

            input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
            labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))

            batch["input_ids"] = self.filter_input_ids(input_ids, input_ids_sentinel)
            batch["labels"] = self.filter_input_ids(input_ids, labels_sentinel)

            # My implementation to solve adding noise tokens after the end of the line
            batch_size = len(batch["input_ids"])
            if self.line_by_line:
                for idx in range(batch_size):

                    one_idx = np.where(batch["input_ids"][idx] == 1)[0]
                    if len(one_idx > 0):
                        end_idx = one_idx[0] + 1
                        batch["input_ids"][idx][end_idx:] = 0
                    zero_idx = np.where(batch["input_ids"][idx] == 0)[0]
                    if len(zero_idx > 0):
                        end_idx = zero_idx[0] + 1
                        batch["input_ids"][idx][end_idx:] = 0

                    zero_idx = np.where(batch["labels"][idx] == 0)[0]
                    if len(zero_idx > 0):
                        end_idx = zero_idx[0] + 1
                        batch["labels"][idx][end_idx:] = -100
                        batch["labels"][idx][end_idx - 1] = 1

                batch["input_ids"] = np.pad(
                    batch["input_ids"],
                    ((0, 0), (0, self.max_length - len(batch["input_ids"][0]))),
                    mode="constant",
                )

            else:
                if batch["input_ids"].shape[-1] != self.input_length:
                    raise ValueError(
                        f"`input_ids` are incorrectly preprocessed. `input_ids` length is {batch['input_ids'].shape[-1]}, but should be {self.target_length}."
                    )

                if batch["labels"].shape[-1] != self.target_length:
                    raise ValueError(
                        f"`labels` are incorrectly preprocessed. `labels` length is {batch['labels'].shape[-1]}, but should be {self.target_length}."
                    )
        else:
            logger.error("Incorrect and not implemented masking strategy.")

        # to check that tokens are correctly proprocessed, one can run `self.tokenizer.batch_decode(input_ids)` and `self.tokenizer.batch_decode(labels)` here...
        batch["decoder_input_ids"] = shift_tokens_right(
            batch["labels"], self.pad_token_id, self.decoder_start_token_id
        )

        return batch

    def generate_equal_amino_masked_indexs(self, sample):
        relative_noise_density = self.noise_density / self.mean_noise_span_length

        masked_indice = np.zeros_like(sample, dtype=bool)
        sample_actual_length = np.nonzero(sample == 1)[0][0]
        sample_aminos = np.unique(sample)
        sample_aminos = sample_aminos[
            np.nonzero(sample_aminos == self.tokenizer.eos_token_id)[0][0] + 1 :
        ]
        sample_masked_aminos = {}
        for masked_tokens_num in range(
            math.ceil(relative_noise_density * sample_actual_length)
        ):
            sample_amino_selected_index = np.random.choice(
                sample_aminos.shape[0], 1, replace=False
            )
            rand_token = sample_aminos[sample_amino_selected_index][0]
            if rand_token not in sample_masked_aminos:
                sample_masked_aminos[rand_token] = 1
            else:
                if len(sample_masked_aminos) != len(sample_aminos):
                    while rand_token in sample_masked_aminos:
                        sample_amino_selected_index = np.random.choice(
                            sample_aminos.shape[0], 1, replace=False
                        )
                        rand_token = sample_aminos[sample_amino_selected_index][0]
                    sample_masked_aminos[rand_token] = 1
                else:
                    while rand_token != min(
                        sample_masked_aminos, key=sample_masked_aminos.get
                    ):
                        sample_amino_selected_index = np.random.choice(
                            sample_aminos.shape[0], 1, replace=False
                        )
                        rand_token = sample_aminos[sample_amino_selected_index][0]
                    sample_masked_aminos[rand_token] += 1
            random_token_indexes = np.nonzero(sample == rand_token)[0]
            random_token_selected_indexes = np.random.choice(
                random_token_indexes.shape[0], 1, replace=False
            )
            chosen_random_token_index = random_token_indexes[
                random_token_selected_indexes
            ]
            masked_indice[chosen_random_token_index] = True
        return masked_indice

    def mask_tokens(
        self, inputs: np.ndarray, special_tokens_mask: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.copy()

        if self.masking_type == "1-gram-amino":
            masked_indices = np.zeros_like(labels, dtype=bool)
            for idx, sample in enumerate(labels):
                end_idx = np.nonzero(sample == 1)[0][0]
                rand_idx = np.random.randint(0, end_idx)
                rand_token = sample[rand_idx]
                masked_indices[idx][sample == rand_token] = True
        elif self.masking_type == "1-gram-amino-eq":
            masked_indices = np.zeros_like(labels, dtype=bool)
            for idx, sample in enumerate(labels):
                rand_token = np.random.randint(3, 28)
                while rand_token not in sample:
                    rand_token = np.random.randint(3, 28)
                masked_indices[idx][sample == rand_token] = True
        elif self.masking_type == "n-gram-tokens-eq":
            # This is my implementation for adjusting the noise density based on the n-gram span
            # relative_noise_density = self.noise_density / self.mean_noise_span_length

            # This is my implementation to make sure we have almost the same probability of masking for each amino acid
            with Pool(16) as p:
                masked_indices_list = p.map(
                    self.generate_equal_amino_masked_indexs, labels
                )
            masked_indices = np.array(masked_indices_list)
            """
            masked_indices = np.zeros_like(labels, dtype=bool)
            for idx, sample in enumerate(labels):
                sample_actual_length = np.nonzero(sample == 1)[0][0]
                sample_aminos = np.unique(sample)
                sample_aminos = sample_aminos[
                    np.nonzero(sample_aminos == self.tokenizer.eos_token_id)[0][0] + 1 :
                ]
                sample_masked_aminos = {}
                for masked_tokens_num in range(
                    math.ceil(relative_noise_density * sample_actual_length)
                ):
                    sample_amino_selected_index = np.random.choice(
                        sample_aminos.shape[0], 1, replace=False
                    )
                    rand_token = sample_aminos[sample_amino_selected_index][0]
                    if rand_token not in sample_masked_aminos:
                        sample_masked_aminos[rand_token] = 1
                    else:
                        if len(sample_masked_aminos) != len(sample_aminos):
                            while rand_token in sample_masked_aminos:
                                sample_amino_selected_index = np.random.choice(
                                    sample_aminos.shape[0], 1, replace=False
                                )
                                rand_token = sample_aminos[sample_amino_selected_index][
                                    0
                                ]
                            sample_masked_aminos[rand_token] = 1
                        else:
                            while rand_token != min(
                                sample_masked_aminos, key=sample_masked_aminos.get
                            ):
                                sample_amino_selected_index = np.random.choice(
                                    sample_aminos.shape[0], 1, replace=False
                                )
                                rand_token = sample_aminos[sample_amino_selected_index][
                                    0
                                ]
                            sample_masked_aminos[rand_token] += 1
                    random_token_indexes = np.nonzero(sample == rand_token)[0]
                    random_token_selected_indexes = np.random.choice(
                        random_token_indexes.shape[0], 1, replace=False
                    )
                    chosen_random_token_index = random_token_indexes[
                        random_token_selected_indexes
                    ]
                    masked_indices[idx][chosen_random_token_index] = True
            """

            # This is my implementation for n-gram masking
            indices_replaced_idx = np.where(masked_indices == True)
            listOfCoordinates = list(
                zip(indices_replaced_idx[0], indices_replaced_idx[1])
            )
            for batch_idx, token_idx in listOfCoordinates:
                n_gram_start_idx = (
                    int(token_idx - (self.mean_noise_span_length / 2)) + 1
                )
                n_gram_end_idx = int(token_idx + (self.mean_noise_span_length / 2)) + 1
                masked_indices[batch_idx][n_gram_start_idx:n_gram_end_idx] = True
        elif self.masking_type == "n-gram-tokens":
            # This is my implementation for adjusting the noise density based on the n-gram span
            relative_noise_density = self.noise_density / self.mean_noise_span_length

            # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
            probability_matrix = np.full(labels.shape, relative_noise_density)
            special_tokens_mask = special_tokens_mask.astype("bool")

            probability_matrix[special_tokens_mask] = 0.0
            masked_indices = np.random.binomial(1, probability_matrix).astype("bool")

            # This is my implementation for n-gram masking
            indices_replaced_idx = np.where(masked_indices == True)
            listOfCoordinates = list(
                zip(indices_replaced_idx[0], indices_replaced_idx[1])
            )
            for batch_idx, token_idx in listOfCoordinates:
                n_gram_start_idx = (
                    int(token_idx - (self.mean_noise_span_length / 2)) + 1
                )
                n_gram_end_idx = int(token_idx + (self.mean_noise_span_length / 2)) + 1
                masked_indices[batch_idx][n_gram_start_idx:n_gram_end_idx] = True

        # This is my code to reconstruct whole input or not
        if self.reconstruct_whole_input == False:
            labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 100% of the time, we replace masked inputs tokens with tokenizer.mask_token ([MASK])
        inputs[masked_indices] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        """
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            np.random.binomial(1, np.full(labels.shape, 0.8)).astype("bool")
            & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        # 10% of the time, we replace masked input tokens with random word
        indices_random = np.random.binomial(1, np.full(labels.shape, 0.5)).astype(
            "bool"
        )
        indices_random &= masked_indices & ~indices_replaced

        random_words = np.random.randint(
            self.tokenizer.vocab_size - len(self.tokenizer.additional_special_tokens),
            size=labels.shape,
            dtype="i4",
        )
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        """

        return inputs, labels

    def create_sentinel_ids(self, mask_indices):
        """
        Sentinel ids creation given the indices that should be masked.
        The start indices of each mask are replaced by the sentinel ids in increasing
        order. Consecutive mask indices to be deleted are replaced with `-1`.
        """
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(
            start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices
        )
        sentinel_ids = np.where(
            sentinel_ids != 0, (len(self.tokenizer) - sentinel_ids), 0
        )
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def filter_input_ids(self, input_ids, sentinel_ids):
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """
        batch_size = input_ids.shape[0]

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed
        input_ids = input_ids_full[input_ids_full >= 0].reshape((batch_size, -1))
        input_ids = np.concatenate(
            [
                input_ids,
                np.full((batch_size, 1), self.tokenizer.eos_token_id, dtype=np.int32),
            ],
            axis=-1,
        )
        return input_ids

    def random_spans_noise_mask(self, length):

        """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .

        Noise mask consisting of random spans of noise tokens.
        The number of noise tokens and the number of noise spans and non-noise spans
        are determined deterministically as follows:
        num_noise_tokens = round(length * noise_density)
        num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
        Spans alternate between non-noise and noise, beginning with non-noise.
        Subject to the above restrictions, all masks are equally likely.

        Args:
            length: an int32 scalar (length of the incoming token sequence)
            noise_density: a float - approximate density of output mask
            mean_noise_span_length: a number

        Returns:
            a boolean tensor with shape [length]
        """

        orig_length = length

        num_noise_tokens = int(np.round(length * self.noise_density))
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = int(np.round(num_noise_tokens / self.mean_noise_span_length))

        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):
            """Partition a sequence of items randomly into non-empty segments.
            Args:
                num_items: an integer scalar > 0
                num_segments: an integer scalar in [1, num_items]
            Returns:
                a Tensor with shape [num_segments] containing positive integers that add
                up to num_items
            """
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            # count length of sub segments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(
            num_nonnoise_tokens, num_noise_spans
        )

        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
            [num_noise_spans * 2],
        )
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length,), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)

        return is_noise[:orig_length]


# def generate_batch_splits(samples_idx: jnp.ndarray, batch_size: int) -> jnp.ndarray:
def generate_batch_splits(samples_idx: np.array, batch_size: int) -> np.array:
    num_samples = len(samples_idx)
    samples_to_remove = num_samples % batch_size

    if samples_to_remove != 0:
        samples_idx = samples_idx[:-samples_to_remove]
    sections_split = num_samples // batch_size
    batch_idx = np.split(samples_idx, sections_split)
    return batch_idx


def write_train_metric(summary_writer, train_metrics, train_time, step, iter_per_sec):
    summary_writer.scalar("train_time", train_time, step)
    summary_writer.scalar("iterations_per_sec", iter_per_sec, step)

    train_metrics = get_metrics(train_metrics)
    for key, vals in train_metrics.items():
        tag = f"train_{key}"
        for i, val in enumerate(vals):
            summary_writer.scalar(tag, val, step - len(vals) + i + 1)


def write_eval_metric(summary_writer, eval_metrics, step):
    for metric_name, value in eval_metrics.items():
        summary_writer.scalar(f"eval_{metric_name}", value, step)


def save_checkpoint(
    model,
    tokenizer,
    repo: Repository,
    save_dir,
    state,
    cur_step: int,
    cur_local_step: int,
    cur_epoch: int,
    cur_samples_idx: int,
    logger,
    with_opt: bool = True,
    push_to_hub: bool = False,
):
    state = jax_utils.unreplicate(state)
    if with_opt:
        logger.info(f"Saving optimizer and training state in {save_dir}...")
        with open(os.path.join(save_dir, "opt_state.msgpack"), "wb") as f:
            f.write(to_bytes(state.opt_state))
        with open(os.path.join(save_dir, "training_state.json"), "w") as f:
            training_state = {
                "step": state.step.item(),
                "local_step": cur_local_step,
                "epoch": cur_epoch,
                "samples_idx": cur_samples_idx,
            }
            json.dump(training_state, f)
    logger.info(
        f'Saving model in {save_dir} {"and pushing it to HF Hub" if push_to_hub else ""}'
    )
    tokenizer.save_pretrained(save_dir)
    model.save_pretrained(
        save_dir,
        params=state.params,
        # push_to_hub=push_to_hub,
        # commit_message=f"Saving weights and logs of step {cur_step}",
    )
    if push_to_hub:
        logger.info(
            "Started to push the current model to the hub async for the current step."
        )
        repo.push_to_hub(
            commit_message=f"Saving weights and logs of step {cur_step}",
            blocking=False,
        )


def restore_checkpoint(load_dir, state, logger):
    logger.info(f"Restoring checkpoint from {load_dir}")
    with open(os.path.join(load_dir, "flax_model.msgpack"), "rb") as f:
        params = from_bytes(state.params, f.read())
    with open(os.path.join(load_dir, "opt_state.msgpack"), "rb") as f:
        opt_state = from_bytes(state.opt_state, f.read())
    with open(os.path.join(load_dir, "training_state.json"), "r") as f:
        training_state = json.load(f)
    step = training_state["step"]
    local_step = training_state["local_step"]
    epoch = training_state["epoch"]
    samples_idx = training_state["samples_idx"]
    logger.info(f"Checkpoint restored at step {step}")
    return (
        state.replace(step=step, params=params, opt_state=opt_state),
        step,
        local_step,
        epoch,
        samples_idx,
    )


class RandomSampler(TorchSampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.
    Arguments:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
    """

    def __init__(
        self,
        data_source,
        seed,
        epoch,
        samples_start_idx,
        replacement=False,
        num_samples=None,
    ):
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.seed = seed
        self.epoch = epoch
        self.samples_start_idx = samples_start_idx

        if not isinstance(self.replacement, bool):
            raise ValueError(
                "replacement should be a boolean value, but got "
                "replacement={}".format(self.replacement)
            )

        if self._num_samples is not None and not replacement:
            raise ValueError(
                "With replacement=False, num_samples should not be specified, "
                "since a random permute will be performed."
            )

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integer "
                "value, but got num_samples={}".format(self.num_samples)
            )

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source) - self.samples_start_idx
        return self._num_samples

    def __iter__(self):
        g_cpu = TorchGenerator()
        g_cpu.manual_seed(self.seed + self.epoch)
        n = len(self.data_source)
        if self.replacement:
            return iter(
                torchRandInt(
                    high=n, size=(self.num_samples,), dtype=torchInt64, generator=g_cpu
                ).tolist(),
            )
        random_batchs_idx = torchRandPerm(n, generator=g_cpu).tolist()
        random_batchs_idx = random_batchs_idx[self.samples_start_idx :]
        return iter(random_batchs_idx)

    def __len__(self):
        return self.num_samples


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    manual_seed(training_args.seed)
    # np.random.seed(training_args.seed)
    # random.seed(training_args.seed)

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        level=logging.INFO,
        datefmt="[%X]",
    )

    # Log on each process the small summary:
    logger = logging.getLogger(__name__)

    # Setup logging, we only want one process per machine to log things on the screen.
    logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)
    if jax.process_index() == 0:
        hfds.utils.logging.set_verbosity_warning()
        hftf.utils.logging.set_verbosity_info()
    else:
        hfds.utils.logging.set_verbosity_error()
        hftf.utils.logging.set_verbosity_error()

    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Handle the repository creation
    if training_args.push_to_hub:
        if training_args.hub_model_id is None:
            repo_name = get_full_repo_name(
                Path(training_args.output_dir).absolute().name,
                token=training_args.hub_token,
            )
        else:
            repo_name = training_args.hub_model_id
        repo = Repository(
            training_args.output_dir,
            clone_from=repo_name,
            use_auth_token=training_args.hub_token,
            # use_auth_token=True,
            private=training_args.private_repo,
        )

    with open(
        os.path.join(training_args.output_dir, "run_parameters_data_args.json"), "w"
    ) as fp:
        data_args_dic = data_args.__dict__
        json.dump(data_args_dic, fp, indent=4)
    with open(
        os.path.join(training_args.output_dir, "run_parameters_model_args.json"), "w"
    ) as fp:
        model_args_dic = model_args.__dict__
        json.dump(model_args_dic, fp, indent=4)
    with open(
        os.path.join(training_args.output_dir, "run_parameters_training_args.json"), "w"
    ) as fp:
        training_args_dic = training_args.__dict__
        training_args_dic.pop("hub_token")
        json.dump(training_args_dic, fp, indent=4)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
        )

        if "validation" not in datasets.keys():
            datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
            )
            datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
            )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = data_args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
        datasets = load_dataset(
            extension, data_files=data_files, cache_dir=model_args.cache_dir
        )

        if "validation" not in datasets.keys():
            datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
            )
            datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
            )
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    logger.info(f"Local TPUs: {jax.local_device_count()}")
    logger.info(f"Global TPUs: {jax.device_count()}")

    # Load pretrained model and tokenizer

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            # use_auth_token=training_args.hub_token,
            use_auth_token=True,
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    tokenizer.mask_token = tokenizer.additional_special_tokens[0]

    if model_args.config_name:
        config = T5Config.from_pretrained(
            model_args.config_name,
            cache_dir=model_args.cache_dir,
            vocab_size=len(tokenizer),
        )
    elif model_args.model_name_or_path:
        config = T5Config.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            # use_auth_token=training_args.hub_token,
            use_auth_token=True,
        )
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    # Initialize our training
    rng = jax.random.PRNGKey(training_args.seed)
    dropout_rngs = jax.random.split(rng, jax.local_device_count())

    if model_args.model_name_or_path:
        model = FlaxT5ForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            seed=training_args.seed,
            dtype=getattr(jnp, model_args.dtype),
            # use_auth_token=training_args.hub_token,
            use_auth_token=True,
        )
    else:
        config.vocab_size = len(tokenizer)
        model = FlaxT5ForConditionalGeneration(
            config, seed=training_args.seed, dtype=getattr(jnp, model_args.dtype)
        )

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    if data_args.masking_strategy == "span":
        if data_args.line_by_line:
            # When using line_by_line, we just tokenize each nonempty line.
            # padding = "max_length" if data_args.pad_to_max_length else False
            padding = "max_length"

            # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
            # Since we make sure that all sequences are of the same length, no attention_mask is needed.
            # def tokenize_function(examples):
            #    return tokenizer(examples[text_column_name], return_attention_mask=False)

            def tokenize_function(examples):
                # Remove empty lines
                examples = [
                    line for line in examples if len(line) > 0 and not line.isspace()
                ]
                return tokenizer(
                    examples,
                    return_special_tokens_mask=False,
                    padding=padding,
                    truncation=True,
                    max_length=max_seq_length,
                )

            tokenized_datasets = datasets.map(
                tokenize_function,
                input_columns=[text_column_name],
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
            )

            # T5-like span masked language modeling will fuse consecutively masked tokens to a single sentinel token.
            # To ensure that the input length is `max_seq_length`, we need to increase the maximum length
            # according to `mlm_probability` and `mean_noise_span_length`. We can also define the label length accordingly.
            expanded_inputs_length, targets_length = compute_input_and_target_lengths(
                inputs_length=max_seq_length,
                noise_density=data_args.mlm_probability,
                mean_noise_span_length=data_args.mean_noise_span_length,
            )
        else:
            # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
            # Since we make sure that all sequences are of the same length, no attention_mask is needed.
            def tokenize_function(examples):
                return tokenizer(
                    examples[text_column_name], return_attention_mask=False
                )

            tokenized_datasets = datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
            )

            # T5-like span masked language modeling will fuse consecutively masked tokens to a single sentinel token.
            # To ensure that the input length is `max_seq_length`, we need to increase the maximum length
            # according to `mlm_probability` and `mean_noise_span_length`. We can also define the label length accordingly.
            expanded_inputs_length, targets_length = compute_input_and_target_lengths(
                inputs_length=max_seq_length,
                noise_density=data_args.mlm_probability,
                mean_noise_span_length=data_args.mean_noise_span_length,
            )

            # Main data processing function that will concatenate all texts from our dataset and generate chunks of expanded_inputs_length.
            def span_group_texts(examples):
                # Concatenate all texts.
                concatenated_examples = {
                    k: list(chain(*examples[k])) for k in examples.keys()
                }
                total_length = len(concatenated_examples[list(examples.keys())[0]])
                # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
                # customize this part to your needs.
                if total_length >= expanded_inputs_length:
                    total_length = (
                        total_length // expanded_inputs_length
                    ) * expanded_inputs_length
                # Split by chunks of max_len.
                result = {
                    k: [
                        t[i : i + expanded_inputs_length]
                        for i in range(0, total_length, expanded_inputs_length)
                    ]
                    for k, t in concatenated_examples.items()
                }
                return result

            # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
            # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
            # might be slower to preprocess.
            #
            # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
            tokenized_datasets = tokenized_datasets.map(
                span_group_texts,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
            )
        # Data collator
        # This one will take care of randomly masking the tokens.
        data_collator = FlaxDataCollatorForT5MLM(
            tokenizer=tokenizer,
            noise_density=data_args.mlm_probability,
            mean_noise_span_length=data_args.mean_noise_span_length,
            input_length=max_seq_length,
            target_length=targets_length,
            pad_token_id=model.config.pad_token_id,
            line_by_line=data_args.line_by_line,
            decoder_start_token_id=model.config.decoder_start_token_id,
            masking_type=data_args.masking_strategy,
            max_length=max_seq_length,
        )
    elif data_args.masking_strategy in [
        "n-gram-tokens",
        "1-gram-amino",
        "1-gram-amino-eq",
        "n-gram-tokens-eq",
    ]:
        if data_args.line_by_line:
            # When using line_by_line, we just tokenize each nonempty line.
            padding = "max_length" if data_args.pad_to_max_length else False

            def tokenize_function(examples):
                # Remove empty lines
                examples = [
                    line for line in examples if len(line) > 0 and not line.isspace()
                ]
                return tokenizer(
                    examples,
                    return_special_tokens_mask=True,
                    padding=padding,
                    truncation=True,
                    max_length=max_seq_length,
                )

            tokenized_datasets = datasets.map(
                tokenize_function,
                input_columns=[text_column_name],
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
            )

        else:
            # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
            # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
            # efficient when it receives the `special_tokens_mask`.
            def tokenize_function(examples):
                return tokenizer(
                    examples[text_column_name], return_special_tokens_mask=True
                )

            tokenized_datasets = datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
            )

            # Main data processing function that will concatenate all texts from our dataset and generate chunks of
            # max_seq_length.
            def n_gram_group_texts(examples):
                # Concatenate all texts.
                concatenated_examples = {
                    k: list(chain(*examples[k])) for k in examples.keys()
                }
                total_length = len(concatenated_examples[list(examples.keys())[0]])
                # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
                # customize this part to your needs.
                if total_length >= max_seq_length:
                    total_length = (total_length // max_seq_length) * max_seq_length
                # Split by chunks of max_len.
                result = {
                    k: [
                        t[i : i + max_seq_length]
                        for i in range(0, total_length, max_seq_length)
                    ]
                    for k, t in concatenated_examples.items()
                }
                return result

            # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
            # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
            # might be slower to preprocess.
            #
            # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
            tokenized_datasets = tokenized_datasets.map(
                n_gram_group_texts,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
            )

        # Data collator
        # This one will take care of randomly masking the tokens.
        data_collator = FlaxDataCollatorForT5MLM(
            tokenizer=tokenizer,
            noise_density=data_args.mlm_probability,
            mean_noise_span_length=data_args.mean_noise_span_length,
            input_length=max_seq_length,
            target_length=None,
            pad_token_id=model.config.pad_token_id,
            decoder_start_token_id=model.config.decoder_start_token_id,
            # masking_type="n-gram-tokens",
            masking_type=data_args.masking_strategy,
            line_by_line=data_args.line_by_line,
            max_length=max_seq_length,
            reconstruct_whole_input=data_args.reconstruct_whole_input,
        )
    else:
        logger.error("The entered masking method is not implemented.")

    # Enable tensorboard only on the master node
    has_tensorboard = is_tensorboard_available()
    if has_tensorboard and jax.process_index() == 0:
        try:
            from flax.metrics.tensorboard import SummaryWriter

            summary_writer = SummaryWriter(log_dir=Path(training_args.output_dir))
        except ImportError as ie:
            has_tensorboard = False
            logger.warning(
                f"Unable to display metrics through TensorBoard because some package are not installed: {ie}"
            )
    else:
        logger.warning(
            "Unable to display metrics through TensorBoard because the package is not installed: "
            "Please run pip install tensorboard to enable."
        )

    # Store some constant
    num_epochs = int(training_args.num_train_epochs)
    # batch size
    batch_size_per_node_per_grad_step = (
        training_args.per_device_train_batch_size
        * jax.local_device_count()
        // training_args.mp_devices
    )
    batch_size_per_node = (
        batch_size_per_node_per_grad_step * training_args.gradient_accumulation_steps
    )
    batch_size_per_step = batch_size_per_node * jax.process_count()
    eval_batch_size_per_node = (
        training_args.per_device_eval_batch_size
        * jax.local_device_count()
        // training_args.mp_devices
    )
    eval_batch_size_per_step = eval_batch_size_per_node * jax.process_count()
    len_train_dataset, len_eval_dataset = tokenized_datasets.length
    steps_per_epoch = (
        len_train_dataset // batch_size_per_node
        if len_train_dataset is not None
        else None
    )
    num_train_steps = (
        steps_per_epoch * num_epochs if steps_per_epoch is not None else None
    )
    num_params = model.num_params

    # train_batch_size = (
    #    int(training_args.per_device_train_batch_size)
    #    * jax.device_count()
    #    * training_args.gradient_accumulation_steps
    # )
    # eval_batch_size = int(training_args.per_device_eval_batch_size) * jax.device_count()

    # num_train_steps = len(tokenized_datasets["train"]) // train_batch_size * num_epochs

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len_train_dataset}")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info(
        f"  Batch size per dp device = {training_args.per_device_train_batch_size}"
    )
    logger.info(f"  Number of devices = {jax.device_count()}")
    logger.info(
        f"  Gradient accumulation steps = {training_args.gradient_accumulation_steps}"
    )
    logger.info(f"  Batch size per update = {batch_size_per_step}")
    logger.info(f"  Model parameters = {num_params:,}")

    # Create learning rate schedule
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=training_args.learning_rate,
        transition_steps=training_args.warmup_steps,
    )
    decay_fn = optax.linear_schedule(
        init_value=training_args.learning_rate,
        end_value=0,
        transition_steps=num_train_steps - training_args.warmup_steps,
    )
    const_fn = optax.constant_schedule(value=training_args.learning_rate)

    if training_args.learning_schedule == "linear_warmup_decay":
        logger.info("Using linear_warmup_decay schedule.")
        learning_schedule_fn = optax.join_schedules(
            schedules=[warmup_fn, decay_fn], boundaries=[training_args.warmup_steps]
        )
    elif training_args.learning_schedule == "linear_warmup_const":
        logger.info("Using linear_warmup_const schedule.")
        learning_schedule_fn = optax.join_schedules(
            schedules=[warmup_fn, const_fn], boundaries=[training_args.warmup_steps]
        )
    else:
        logger.error("The learning rate schedule you choose in not implemented.")

    # We use Optax's "masking" functionality to not apply weight decay
    # to bias and LayerNorm scale parameters. decay_mask_fn returns a
    # mask boolean with the same structure as the parameters.
    # The mask is True for parameters that should be decayed.
    def decay_mask_fn(params):
        flat_params = traverse_util.flatten_dict(params)
        flat_mask = {
            path: (
                path[-1] != "bias"
                and path[-2:]
                not in [("layer_norm", "scale"), ("final_layer_norm", "scale")]
            )
            for path in flat_params
        }
        return traverse_util.unflatten_dict(flat_mask)

    # create adam optimizer
    if training_args.optim == "adafactor":
        # We use the default parameters here to initialize adafactor,
        # For more details about the parameters please check https://github.com/deepmind/optax/blob/ed02befef9bf81cbbf236be3d2b0e032e9ed4a40/optax/_src/alias.py#L74
        logger.info("Using adafactor optimizer.")
        optimizer = optax.adafactor(
            learning_rate=learning_schedule_fn,
        )
    elif training_args.optim == "sm3":
        logger.info("Using SM3 optimizer.")
        optimizer = optax.chain(
            optax.scale_by_sm3(
                b1=training_args.optim_beta1,
                b2=training_args.optim_beta2,
                eps=training_args.optim_epsilon,
            ),
            optax.scale_by_schedule(lambda count: -1 * learning_schedule_fn(count)),
        )
    elif training_args.optim == "shampoo":
        logger.info("Using distributed shampoo optimizer.")
        # https://github.com/google-research/google-research/tree/master/scalable_shampoo
        optimizer = distributed_shampoo.distributed_shampoo(
            learning_rate=learning_schedule_fn,
            block_size=training_args.shampoo_block_size,
            beta1=training_args.optim_beta1,
            beta2=training_args.optim_beta2,
            diagonal_epsilon=1e-10,
            matrix_epsilon=training_args.optim_epsilon,
            weight_decay=training_args.optim_weight_decay,
            start_preconditioning_step=training_args.shampoo_start_preconditioning_step,
            preconditioning_compute_steps=training_args.shampoo_preconditioning_compute_steps,
            statistics_compute_steps=1,
            best_effort_shape_interpretation=True,
            graft_type=distributed_shampoo.GraftingType.RMSPROP_NORMALIZED,  # GraftingType.SGD
            nesterov=training_args.shampoo_nesterov,
            exponent_override=0,
            # Pass pmap 'batch axis name' in pmap mode.
            batch_axis_name="batch",
            ### Only set following 3 params in pjit/spmd mode.
            ### WARNING: Experimental
            statistics_partition_spec=None,  # PartitionSpec(None, "batch", None),
            preconditioner_partition_spec=None,  # PartitionSpec("batch", None, None),
            num_devices_for_pjit=None,  # training_args.dp_devices,
            shard_optimizer_states=False,  # True
            ###
            ### Experimental memory reduction mode
            best_effort_memory_usage_reduction=training_args.shampoo_optim_quantized,
            ###
            inverse_failure_threshold=0.1,
            moving_average_for_momentum=training_args.shampoo_moving_average_for_momentum,
            skip_preconditioning_dim_size_gt=training_args.shampoo_skip_preconditioning_dim_size_gt,
            clip_by_scaled_gradient_norm=None,
            precision=lax.Precision.HIGHEST,
        )
    else:
        logger.info("Using adamw optimizer.")
        optimizer = optax.adamw(
            learning_rate=learning_schedule_fn,
            b1=training_args.optim_beta1,
            b2=training_args.optim_beta2,
            weight_decay=training_args.optim_weight_decay,
            mask=decay_mask_fn,
        )

    if training_args.gradient_accumulation_steps > 1:
        optimizer = optax.MultiSteps(
            optimizer, training_args.gradient_accumulation_steps
        )
    grad_accum_steps = training_args.gradient_accumulation_steps

    # get PartitionSpec for model params (required to be a dict)
    param_spec = set_partitions(model.params)

    # convert params to frozen dict
    model._params = freeze(model.params)

    # get PartitionSpec for optimizer state
    def get_opt_state_spec_and_shape(param_spec):
        # get opt_state shape without actual init
        opt_state_shape = jax.eval_shape(optimizer.init, model.params)

        if training_args.optim == "adam":

            def _opt_state_spec_per_leaf(x):
                if isinstance(x, FrozenDict):
                    # variables with same structure as params
                    return param_spec
                else:
                    # other variables such as count
                    return None

            opt_state_spec = jax.tree_map(
                _opt_state_spec_per_leaf,
                opt_state_shape,
                # return None spec for empty elements
                is_leaf=lambda x: isinstance(x, (FrozenDict, optax.EmptyState)),
            )

        elif training_args.optim == "adafactor":
            # factorized state must be replicated (rank different than params)
            opt_state_spec = None

        elif training_args.optim == "distributed_shampoo":
            opt_state_spec = opt_fn.pspec_fn(
                params=model.params,
                params_partition_spec=param_spec,
                partition_spec_for_statistics=PartitionSpec(None, "dp", None),
            )
        else:
            raise NotImplementedError
        return opt_state_spec, opt_state_shape

    opt_state_spec, opt_state_shape = get_opt_state_spec_and_shape(param_spec)

    # create a mesh
    mesh_shape = (training_args.dp_devices, training_args.mp_devices)
    devices = np.asarray(jax.devices()).reshape(*mesh_shape)
    mesh = maps.Mesh(devices, ("dp", "mp"))
    logger.info(f"  Mesh shape: {mesh_shape}")

    # define state spec
    state_spec = TrainState(
        params=param_spec,
        opt_state=opt_state_spec,
        dropout_rng=None,
        step=None,
        epoch=None,
        train_time=None,
        train_samples=None,
        apply_fn=train_fn,
        tx=optimizer,
    )

    # init params if not available yet
    def maybe_init_params(params):
        if model_args.model_name_or_path:
            # model params are correctly loaded
            return params
        else:
            # params have not been initialized yet
            return model.init_weights()

    """
    # Setup train state
    state = train_state.TrainState.create(
        apply_fn=model.__call__, params=model.params, tx=optimizer
    )
    """
    with maps.mesh(mesh.devices, mesh.axis_names):
        logger.info("  Creating state")
        if training_args.resume_from_checkpoint:
            (
                state,
                resume_step,
                resume_local_step,
                resume_epoch,
                resume_samples_idx,
            ) = restore_checkpoint(
                # training_args.resume_from_checkpoint, state, logger
                training_args.output_dir,
                state,
                logger,
            )
        else:

            def init_state(params):
                return TrainState.create(
                    apply_fn=train_fn,
                    tx=optimizer,
                    params=maybe_init_params(params),
                    dropout_rng=dropout_rng,
                )

            state = pjit(
                init_state,
                in_axis_resources=(param_spec,)
                if model_args.model_name_or_path
                else None,
                out_axis_resources=state_spec,
                donate_argnums=(0,),
            )(model.params if model_args.model_name_or_path else None)

            resume_step = 0
            resume_epoch = 0
            resume_samples_idx = 0
            resume_local_step = 0

    # Define gradient update step fn
    def train_step(state, batch, dropout_rng):
        dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)

        def loss_fn(params):
            labels = batch.pop("labels")

            logits = state.apply_fn(
                **batch, params=params, dropout_rng=dropout_rng, train=True
            )[0]

            # compute loss
            # loss = optax.softmax_cross_entropy(
            #    logits, onehot(labels, logits.shape[-1])
            # ).mean()
            # compute loss, ignore padded input tokens
            label_mask = jnp.where(labels > 0, 1.0, 0.0)
            loss = (
                optax.softmax_cross_entropy(logits, onehot(labels, logits.shape[-1]))
                * label_mask
            )

            # take average
            loss = loss.sum() / label_mask.sum()

            return loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grad = grad_fn(state.params)
        grad = jax.lax.pmean(grad, "batch")
        new_state = state.apply_gradients(grads=grad)

        # metrics = jax.lax.pmean(
        #    {"loss": loss, "learning_rate": learning_schedule_fn(state.step)},
        #    axis_name="batch",
        # )
        metrics = jax.lax.pmean(
            {
                "loss": loss,
                "learning_rate": learning_schedule_fn(state.step // grad_accum_steps),
            },
            axis_name="batch",
        )

        return new_state, metrics, new_dropout_rng

    # Create parallel version of the train step
    p_train_step = jax.pmap(train_step, "batch", donate_argnums=(0,))

    # Define eval fn
    def eval_step(params, batch):
        labels = batch.pop("labels")

        logits = model(**batch, params=params, train=False)[0]

        """
        # compute loss
        loss = optax.softmax_cross_entropy(logits, onehot(labels, logits.shape[-1]))

        # compute accuracy
        accuracy = jnp.equal(jnp.argmax(logits, axis=-1), labels)

        # summarize metrics
        metrics = {"loss": loss.mean(), "accuracy": accuracy.mean()}
        metrics = jax.lax.pmean(metrics, axis_name="batch")
        """
        # compute loss, ignore padded input tokens
        label_mask = jnp.where(labels > 0, 1.0, 0.0)
        loss = (
            optax.softmax_cross_entropy(logits, onehot(labels, logits.shape[-1]))
            * label_mask
        )

        # compute accuracy
        accuracy = jnp.equal(jnp.argmax(logits, axis=-1), labels) * label_mask

        # summarize metrics
        metrics = {
            "loss": loss.sum(),
            "accuracy": accuracy.sum(),
            "normalizer": label_mask.sum(),
        }
        metrics = jax.lax.psum(metrics, axis_name="batch")

        return metrics

    p_eval_step = jax.pmap(eval_step, "batch", donate_argnums=(0,))

    # Replicate the train state on each device
    state = jax_utils.replicate(state)

    # My code replacement to speedup dataloading
    # num_train_samples = len(tokenized_datasets["train"])
    tokenized_datasets.set_format(
        type="torch", columns=["input_ids", "attention_mask", "special_tokens_mask"]
    )

    """
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(tokenized_datasets)}")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed and grad_accum) = {train_batch_size}"
    )
    # logger.info(f"  Total optimization steps = {total_train_steps}")
    """

    if not training_args.skip_memory_metrics:
        server = jax.profiler.start_server(9999)

    train_time = 0
    epochs = tqdm(
        range(resume_epoch, num_epochs),
        desc="Epoch ... ",
        position=0,
        initial=resume_epoch,
        total=num_epochs,
    )
    for epoch in epochs:
        # ======================== Training ================================
        train_start = time.time()
        train_metrics = []

        # Create sampling rng
        rng, input_rng = jax.random.split(rng)

        """
        # Generate an epoch by shuffling sampling indices from the train dataset
        num_train_samples = len(tokenized_datasets["train"])
        # train_samples_idx = jax.random.permutation(input_rng, jnp.arange(num_train_samples))
        train_samples_idx = np.random.permutation(num_train_samples)
        train_batch_idx = generate_batch_splits(train_samples_idx, train_batch_size)

        # Gather the indexes for creating the batch and do a training step
        for step, batch_idx in enumerate(
                tqdm(train_batch_idx, desc="Training...", position=1)
        ):
            samples = [tokenized_datasets["train"][int(idx)] for idx in batch_idx]

            model_inputs = data_collator(samples)
        """

        train_data_loader = DataLoader(
            tokenized_datasets["train"],
            batch_size=train_batch_size,
            collate_fn=data_collator,
            num_workers=data_args.data_loader_num_workers,
            shuffle=False,
            prefetch_factor=data_args.data_loader_prefetch_factor,
            drop_last=True,
            sampler=RandomSampler(
                data_source=tokenized_datasets["train"],
                seed=training_args.seed,
                epoch=epoch,
                samples_start_idx=resume_samples_idx if epoch == resume_epoch else 0,
            ),
        )
        # total_num_training_batchs = len(train_data_loader)
        total_num_training_batchs = (
            len(train_data_loader)
            if epoch != resume_epoch
            else len(tokenized_datasets["train"]) // train_batch_size
        )
        # num_eval_samples = len(tokenized_datasets["validation"])
        valid_data_loader = DataLoader(
            tokenized_datasets["validation"],
            batch_size=eval_batch_size,
            collate_fn=data_collator,
            num_workers=data_args.data_loader_num_workers,
            shuffle=False,
            prefetch_factor=data_args.data_loader_prefetch_factor,
            drop_last=True,
        )

        for step, model_inputs in tqdm(
            enumerate(
                train_data_loader, resume_local_step if epoch == resume_epoch else 0
            ),
            desc="Training...",
            position=1,
            initial=resume_local_step,
            total=total_num_training_batchs,
        ):
            # model_inputs = sample
            # step = batch_idx

            # cur_step = epoch * (num_train_samples // train_batch_size) + step
            cur_step = epoch * total_num_training_batchs + step
            if cur_step < resume_step:
                continue

            # Model forward
            model_inputs = shard(model_inputs.data)
            state, train_metric, dropout_rngs = p_train_step(
                state, model_inputs, dropout_rngs
            )
            train_metrics.append(train_metric)

            # if cur_step % training_args.logging_steps == 0 and cur_step > 0:
            if (
                cur_step % training_args.logging_steps * grad_accum_steps == 0
                and cur_step > 0
            ):
                # Save metrics
                train_metric = jax_utils.unreplicate(train_metric)
                train_time += time.time() - train_start
                iter_per_sec = (cur_step - resume_step) / (time.time() - train_start)
                if has_tensorboard and jax.process_index() == 0:
                    write_train_metric(
                        summary_writer,
                        train_metrics,
                        train_time,
                        cur_step,
                        iter_per_sec,
                    )

                """
                epochs.write(
                    f"Step... ({cur_step} | Loss: {train_metric['loss'].mean()}, Learning Rate: {train_metric['learning_rate'].mean()})"
                )
                """
                epochs.write(
                    f"Step... ({cur_step} | Loss: {train_metric['loss']}, Learning Rate: {train_metric['learning_rate']})"
                )

                train_metrics = []

            # if cur_step % training_args.eval_steps == 0 and cur_step > 0:
            if (
                cur_step % training_args.eval_steps * grad_accum_steps == 0
                and cur_step > 0
            ):
                # ======================== Evaluating ==============================
                """
                num_eval_samples = len(tokenized_datasets["validation"])
                eval_samples_idx = jnp.arange(num_eval_samples)
                eval_batch_idx = generate_batch_splits(
                    eval_samples_idx, eval_batch_size
                )

                eval_metrics = []
                for i, batch_idx in enumerate(
                    tqdm(eval_batch_idx, desc="Evaluating ...", position=2)
                ):
                    samples = [
                        tokenized_datasets["validation"][int(idx)] for idx in batch_idx
                    ]
                    model_inputs = data_collator(samples)
                """
                eval_metrics = []
                for eval_batch_idx, model_inputs in tqdm(
                    enumerate(valid_data_loader),
                    desc="Evaluating...",
                    position=2,
                    total=len(valid_data_loader),
                ):
                    # Model forward
                    model_inputs = shard(model_inputs.data)
                    metrics = p_eval_step(state.params, model_inputs)
                    eval_metrics.append(metrics)

                """
                # get eval metrics
                eval_metrics = get_metrics(eval_metrics)
                eval_metrics = jax.tree_map(jnp.mean, eval_metrics)
                """
                # normalize eval metrics
                eval_metrics = get_metrics(eval_metrics)
                eval_metrics = jax.tree_map(jnp.sum, eval_metrics)
                eval_normalizer = eval_metrics.pop("normalizer")
                eval_metrics = jax.tree_map(lambda x: x / eval_normalizer, eval_metrics)

                # Update progress bar
                epochs.write(
                    f"Step... ({cur_step} | Loss: {eval_metrics['loss']}, Acc: {eval_metrics['accuracy']})"
                )

                # Save metrics
                if has_tensorboard and jax.process_index() == 0:
                    write_eval_metric(summary_writer, eval_metrics, cur_step)

            # if cur_step % training_args.save_steps == 0 and cur_step > 0:
            if (
                cur_step % training_args.save_steps * grad_accum_steps == 0
                and cur_step > 0
            ):
                # save checkpoint after each epoch and push checkpoint to the hub
                if jax.process_index() == 0:
                    """
                    params = jax.device_get(jax.tree_map(lambda x: x[0], state.params))
                    model.save_pretrained(training_args.output_dir, params=params)
                    tokenizer.save_pretrained(training_args.output_dir)
                    if training_args.push_to_hub:
                        logger.info(
                            "Started to push the current model to the hub async for the current step."
                        )
                    repo.push_to_hub(
                        commit_message=f"Saving weights and logs of step {cur_step}",
                        blocking=False,
                    )
                    """
                    save_checkpoint(
                        model,
                        tokenizer,
                        repo,
                        training_args.output_dir,
                        state,
                        cur_step,
                        step,
                        epoch,
                        step * train_batch_size
                        if epoch != resume_epoch
                        else step * train_batch_size + resume_samples_idx,
                        logger,
                        with_opt=True,
                        push_to_hub=training_args.push_to_hub,
                    )

    # Eval after training
    if training_args.do_eval:
        """
        num_eval_samples = len(tokenized_datasets["validation"])
        eval_samples_idx = jnp.arange(num_eval_samples)
        eval_batch_idx = generate_batch_splits(eval_samples_idx, eval_batch_size)

        eval_metrics = []
        for i, batch_idx in enumerate(
            tqdm(eval_batch_idx, desc="Evaluating ...", position=2)
        ):
            samples = [tokenized_datasets["validation"][int(idx)] for idx in batch_idx]
            model_inputs = data_collator(samples)
        """
        eval_metrics = []
        for eval_batch_idx, model_inputs in tqdm(
            enumerate(valid_data_loader),
            desc="Evaluating...",
            position=2,
            total=len(valid_data_loader),
        ):
            # Model forward
            model_inputs = shard(model_inputs.data)
            metrics = p_eval_step(state.params, model_inputs)
            eval_metrics.append(metrics)

        """
        # get eval metrics
        eval_metrics = get_metrics(eval_metrics)
        eval_metrics = jax.tree_map(
            lambda metric: jnp.mean(metric).item(), eval_metrics
        )
        """
        # normalize eval metrics
        eval_metrics = get_metrics(eval_metrics)
        eval_metrics = jax.tree_map(lambda metric: jnp.sum(metric).item(), eval_metrics)
        eval_normalizer = eval_metrics.pop("normalizer")
        eval_metrics = jax.tree_map(lambda x: x / eval_normalizer, eval_metrics)
        try:
            perplexity = math.exp(eval_metrics["loss"])
        except OverflowError:
            perplexity = float("inf")
        eval_metrics["perplexity"] = perplexity

        if jax.process_index() == 0:
            eval_metrics = {
                f"eval_{metric_name}": value
                for metric_name, value in eval_metrics.items()
            }
            path = os.path.join(training_args.output_dir, "eval_results.json")
            with open(path, "w") as f:
                json.dump(eval_metrics, f, indent=4, sort_keys=True)

    if jax.process_index() == 0:
        save_checkpoint(
            model,
            tokenizer,
            repo,
            training_args.output_dir,
            state,
            cur_step,
            step,
            epoch,
            step * train_batch_size
            if epoch != resume_epoch
            else step * train_batch_size + resume_samples_idx,
            logger,
            with_opt=True,
            push_to_hub=training_args.push_to_hub,
        )


if __name__ == "__main__":
    main()
