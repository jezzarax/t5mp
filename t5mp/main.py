import logging

import click

from t5mp.configuration import generate_configuration
from t5mp.tokenizer import train_tokenizer
from t5mp.model import train_model


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)


@click.group()
@click.version_option()
def cli():
    pass


cli.add_command(generate_configuration)
cli.add_command(train_tokenizer)

## python run_t5_mlm_flax.py \
##  --output_dir="./norwegian-t5-base" \
##  --model_type="t5" \
##  --config_name="./norwegian-t5-base" \
##  --tokenizer_name="./norwegian-t5-base" \
##  --dataset_name="oscar" \
##  --dataset_config_name="unshuffled_deduplicated_no" \
##  --max_seq_length="512" \
##  --per_device_train_batch_size="32" \
##  --per_device_eval_batch_size="32" \
##  --adafactor \
##  --learning_rate="0.005" \
##  --weight_decay="0.001" \
##  --warmup_steps="2000" \
##  --overwrite_output_dir \
##  --logging_steps="500" \
##  --save_steps="10000" \
##  --eval_steps="2500" \
##  --push_to_hub

cli.add_command(train_model)

if __name__ == "__main__":
    cli()
