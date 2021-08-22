from functools import partial
from typing import List, Tuple

import torch
import torch.nn.functional as F

from src.bpe_tokenizer import BPETokenizer


class AddressParsingDataset(torch.utils.data.Dataset):

    def __init__(self, addresses, tags):
        self.addresses = addresses
        self.tags = tags

    def __len__(self):
        return len(self.addresses)

    def __getitem__(self, index):
        return self.addresses[index], self.tags[index]


def collate_fn(
        batch: List[List[Tuple[str, List[str]]]],
        tokenizer: BPETokenizer,
) -> torch.tensor:
    """Takes a batch of data, tokenizes and truncates/pads it, stacks into tensor.

    Args:
        batch: batch of data.
        tokenizer to use for tokenization.
        max_len: maximum length of a single input sample.

    Returns:
        Stacked prepared input.
    """
    prepared_sents = []
    for sent, _ in batch:
    # for sent in batch:
        sent = tokenizer.tokenize(sent)
        if len(sent) > tokenizer.max_len:
            sent = sent[:tokenizer.max_len]
        else:
            sent = F.pad(sent, (0, tokenizer.max_len-len(sent)), mode='constant', value=0)
        prepared_sents.append(sent)
    return torch.stack(prepared_sents)


def build_data_loader(
        X: List[str],
        y: List[List[str]],
        batch_size: int,
        tokenizer: BPETokenizer,
) -> torch.utils.data.DataLoader:
    """Build data loader.

    Args:
        X: list of textual data.
        y: list of tags (TODO).
        batch_size: how many samples in a single batch.
        tokenizer: tokenizer to use for tokenization.
        max_len: maximum length of a single input sample.

    Returns:
        An iterable dataloader yielding batches of data.
    """
    dataset = AddressParsingDataset(X, y)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
    )
    return dataloader
