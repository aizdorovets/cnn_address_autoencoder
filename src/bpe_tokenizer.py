from typing import List

import youtokentome as yttm
import torch


class BPETokenizer():

    def __init__(self, path, max_len=512):
        self.bpe_model = self._load(path)
        self.max_len = max_len

    def _load(self, path):
        return yttm.BPE(model=path)

    def tokenize(self, input: str) -> torch.tensor:
        """Tokenizes input string, turns tokens to corresponding ids and makes a tensor out of it.

        Args:
            input: input string.

        Returns:
            Torch tensor with tokens' ids.
        """
        ids = self.bpe_model.encode(input, output_type=yttm.OutputType.ID)
        return torch.tensor(ids, dtype=torch.long)

    def detokenize(self, input: List[List[int]]) -> List[str]:
        return self.bpe_model.decode(input, ignore_ids=[0])
