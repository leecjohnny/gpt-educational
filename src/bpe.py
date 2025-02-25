import os
import pickle
from collections import Counter
from functools import lru_cache
from typing import Dict, List, Protocol, Tuple

import tiktoken

# Type definitions for the vocab maps
T_VocabToID = Dict[Tuple[int, ...], int]
T_IDToVocab = Dict[int, Tuple[int, ...]]


class Tokenizer(Protocol):
    """Protocol defining the interface for tokenizers."""

    def encode(self, text: str) -> List[int]:
        """Convert a string to token IDs."""
        ...

    def decode(self, token_ids: List[int]) -> str:
        """Convert token IDs back to a string."""
        ...

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size of the tokenizer."""
        ...


class TiktokenTokenizer:
    """Wrapper for tiktoken's tokenizers."""

    def __init__(self, model_name: str = "gpt2"):
        """Initialize with a tiktoken encoding for the specified model."""
        self.encoding = tiktoken.encoding_for_model(model_name)

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs using tiktoken."""
        return self.encoding.encode(text)

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text using tiktoken."""
        return self.encoding.decode(token_ids)

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size of the tiktoken encoding."""
        return self.encoding.n_vocab


class BPETokenizer:
    """Simple BPE tokenizer implementation."""

    def __init__(self, max_vocab_size: int = 2000):
        """Initialize with a maximum vocabulary size."""
        self.max_vocab_size = max_vocab_size
        self.vocab_map: T_VocabToID = {}
        self.ids_to_pairs: T_IDToVocab = {}
        # Cache for frequently used encoding results
        self._encoding_cache: Dict[bytes, List[int]] = {}

    def _get_stats(self, token_ids: List[int]) -> Counter[Tuple[int, int]]:
        """Count frequency of adjacent token pairs."""
        int_pairs = zip(token_ids, token_ids[1:])
        return Counter(int_pairs)

    def _merge(
        self, existing_ids: List[int], pair: Tuple[int, int], new_id: int
    ) -> List[int]:
        """Replace all occurrences of a token pair with a new token ID."""
        new_ids = []
        assert len(existing_ids) > 1, "Not enough pairs to merge"
        idx_to_skip: int | None = None
        for idx, (i0, i1) in enumerate(zip(existing_ids, existing_ids[1:] + [None])):
            if idx == idx_to_skip:
                continue
            elif i0 == pair[0] and i1 == pair[1]:
                new_ids.append(new_id)
                idx_to_skip = idx + 1
                continue
            new_ids.append(i0)
        return new_ids

    def train(self, input_ids: List[int]) -> None:
        """Train the BPE tokenizer on the given input data."""
        # Initialize vocab with byte tokens
        self.vocab_map = {(i,): i for i in range(256)}
        current_sequence: List[int] = input_ids
        merges_completed: int = 0

        # Determine how many merges to perform
        max_merges = max(0, self.max_vocab_size - 256)

        # Clear any previous cache
        self._encoding_cache = {}

        # Merge the most frequent pairs until we reach our vocabulary limit
        while merges_completed < max_merges:
            pair_stats = self._get_stats(current_sequence)
            if not pair_stats:
                break

            most_frequent_pair: Tuple[int, int] = pair_stats.most_common(1)[0][0]
            freq = pair_stats[most_frequent_pair]

            # Stop if no more compression is possible
            if freq == 1:
                break

            # Add the new token to the vocabulary
            next_id = len(self.vocab_map)
            self.vocab_map[most_frequent_pair] = next_id

            # Update the sequence by merging all instances of the pair
            current_sequence = self._merge(
                current_sequence, most_frequent_pair, next_id
            )
            merges_completed += 1

        print(
            f"Initial sequence length: {len(input_ids)} | "
            f"Final sequence length: {len(current_sequence)} | "
            f"Vocab size: {len(self.vocab_map)} | "
            f"Compressed ratio: {len(input_ids) / len(current_sequence)}"
        )

        # Build the reverse mapping for decoding
        self.ids_to_pairs = {v: k for k, v in self.vocab_map.items()}

    def save(self, path: str) -> None:
        """Save the tokenizer to a file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        # Convert tuple keys to strings for serialization
        serializable_vocab = {str(k): v for k, v in self.vocab_map.items()}

        with open(path, "wb") as f:
            pickle.dump(
                {
                    "max_vocab_size": self.max_vocab_size,
                    "vocab_map": serializable_vocab,
                },
                f,
            )

        print(f"Tokenizer saved to {path}")

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """Load a tokenizer from a file."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        # Convert string keys back to tuples
        vocab_map = {}
        for k, v in data["vocab_map"].items():
            key = eval(k)  # Safe since we created these strings ourselves
            vocab_map[key] = v

        tokenizer = cls(max_vocab_size=data["max_vocab_size"])
        tokenizer.vocab_map = vocab_map
        tokenizer.ids_to_pairs = {v: k for k, v in vocab_map.items()}
        return tokenizer

    @lru_cache(maxsize=1024)
    def _get_bytes_from_token_tree(self, token: int) -> bytes:
        """Recursively expand a token into its constituent bytes."""
        if token < 256:
            return bytes([token])
        pair = self.ids_to_pairs[token]
        return b"".join(self._get_bytes_from_token_tree(b) for b in pair)

    def decode_bytes(self, token_ids: List[int]) -> bytes:
        """Decode token IDs to bytes."""
        # More efficient implementation using bytearray
        result = bytearray()
        for token in token_ids:
            result.extend(self._get_bytes_from_token_tree(token))
        return bytes(result)

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to a UTF-8 string."""
        text_bytes = self.decode_bytes(token_ids)
        return text_bytes.decode("utf-8", errors="replace")

    def encode_bytes(self, text_bytes: bytes) -> List[int]:
        vocab_map_minus_single_bytes = {k: v for k, v in self.vocab_map.items()}
        ids = list(text_bytes)
        while len(ids) >= 2:
            pairs = self._get_stats(ids)
            pair = min(
                pairs, key=lambda p: vocab_map_minus_single_bytes.get(p, float("inf"))
            )
            if pair not in vocab_map_minus_single_bytes:
                break
            idx = vocab_map_minus_single_bytes[pair]
            ids = self._merge(ids, pair, idx)
        return ids

    def encode(self, text: str) -> List[int]:
        """Encode a string to token IDs."""
        return self.encode_bytes(text.encode("utf-8"))

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size of the tokenizer."""
        return len(self.vocab_map)
