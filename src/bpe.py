from collections import Counter
from typing import Dict, List, Tuple

## add custom type for the vocab map
T_VocabToID = Dict[Tuple[int] | Tuple[int, int], int]
T_IDToVocab = Dict[int, Tuple[int] | Tuple[int, int]]


class BPETokenizer:
    def __init__(self, max_vocab_size: int | None = None):
        self.max_vocab_size = max_vocab_size
        self.vocab_map: T_VocabToID = {}
        self.ids_to_pairs: T_IDToVocab = {}

    def _get_stats(self, token_ids: List[int]) -> Counter[Tuple[int, int]]:
        int_pairs = zip(token_ids, token_ids[1:])
        return Counter(int_pairs)

    def _merge(
        self, existing_ids: List[int], pair: Tuple[int, int], new_id: int
    ) -> List[int]:
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
        self.vocab_map = {(i,): i for i in range(256)}
        current_sequence: List[int] = input_ids
        merges_completed: int = 0
        if self.max_vocab_size:
            max_merges = max(0, self.max_vocab_size - 256)
        else:
            max_merges = 2000
        while merges_completed < max_merges:
            pair_stats = self._get_stats(current_sequence)
            if not pair_stats:
                break
            most_frequent_pair: Tuple[int, int] = pair_stats.most_common(1)[0][0]
            freq = pair_stats[most_frequent_pair]
            if freq == 1:
                break
            next_id = len(self.vocab_map)
            self.vocab_map[most_frequent_pair] = next_id
            new_sequence = self._merge(current_sequence, most_frequent_pair, next_id)
            current_sequence = new_sequence
            merges_completed += 1

        print(
            f"Initial sequence length: {len(input_ids)} | Final sequence length: {len(current_sequence)} | Vocab size: {len(self.vocab_map)} | Compressed ratio: {len(input_ids) / len(current_sequence)}"
        )
        self.ids_to_pairs = {v: k for k, v in self.vocab_map.items()}

    def _get_bytes_from_token_tree(self, token: int) -> bytes:
        if token < 256:
            return bytes([token])
        pair = self.ids_to_pairs[token]
        return b"".join(self._get_bytes_from_token_tree(b) for b in pair)

    def decode_bytes(self, token_ids: List[int]) -> bytes:
        text_bytes = b""
        for token in token_ids:
            text_bytes += self._get_bytes_from_token_tree(token)
        return text_bytes

    def decode(self, token_ids: List[int]) -> str:
        text_bytes = self.decode_bytes(token_ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

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
        return self.encode_bytes(text.encode("utf-8"))
