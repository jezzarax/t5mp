from transformers import T5Config
import click
import os


@click.command("config")
@click.option("--name", default="t5mumo")
@click.option("--vocab-size", type=int, default=32000)
def generate_configuration(name, vocab_size):
    os.makedirs(f"./{name}", exist_ok=True)
    config = T5Config.from_pretrained("google/t5-v1_1-base", vocab_size=vocab_size)
    config.save_pretrained(f"./{name}")
