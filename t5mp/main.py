import logging

import click

from t5mp.configuration import generate_configuration
from t5mp.tokenizer import train_tokenizer
from t5mp.run_t5_mlm_flax import train_model


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)


@click.group()
@click.version_option()
def cli():
    pass


cli.add_command(generate_configuration)
cli.add_command(train_tokenizer)
cli.add_command(train_model)

if __name__ == "__main__":
    cli()
