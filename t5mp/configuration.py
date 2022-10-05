from transformers import T5Config
import click


@click.command("config")
@click.option("--name", default="t5mumo")
@click.option("--vocab-size", type=int, default=32000)
def generate_configuration(name, vocab_size):
    config = T5Config.from_pretrained("google/t5-v1_1-base", vocab_size=vocab_size)
    config.save_pretrained(f"./{name}")
