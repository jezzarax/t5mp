import datasets
import click
import os

from t5mp.t5_tokenizer_model import SentencePieceUnigramTokenizer


@click.command("tokenizer")
@click.option("--vocab-size", type=int, default=32000)
@click.option("--name", default="t5mumo")
@click.option("--dataset-name", default="wikitext")
@click.option("--dataset-config-name", default="wikitext-103-v1")
def train_tokenizer(vocab_size, name, dataset_name, dataset_config_name):
    os.makedirs(f"./{name}", exist_ok=True)
    input_sentence_size = None

    # Initialize a dataset
    dataset = datasets.load_dataset(
        dataset_name, name=dataset_config_name, split="train"
    )

    tokenizer = SentencePieceUnigramTokenizer(
        unk_token="<unk>", eos_token="</s>", pad_token="<pad>"
    )

    # Build an iterator over this dataset
    def batch_iterator(input_sentence_size=None):
        if input_sentence_size is None:
            input_sentence_size = len(dataset)
        batch_length = 100
        for i in range(0, input_sentence_size, batch_length):
            yield dataset[i : i + batch_length]["text"]

    # Train tokenizer
    tokenizer.train_from_iterator(
        iterator=batch_iterator(input_sentence_size=input_sentence_size),
        vocab_size=vocab_size,
        show_progress=True,
    )

    # Save files to disk
    tokenizer.save(f"./{name}/tokenizer.json")
