from typing import Literal
from torch.utils.data import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


Split = Literal["train", "test", "eval"]


class MinRLDataset(Dataset):
    def __init__(self, split: Split, tokenizer: PreTrainedTokenizerBase | None = None):
        self.split = split
        self.tokenizer = tokenizer
