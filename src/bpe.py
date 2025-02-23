from collections import Counter
from typing import Dict, List, Tuple


def get_stats(token_ids: List[int]) -> Counter[Tuple[int, int]]:
    int_pairs = zip(token_ids, token_ids[1:])
    return Counter(int_pairs)


def merge(existing_ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
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


def train_bpe_vocab(
    input_ids: List[int], max_vocab_size: int | None = None
) -> Dict[Tuple[int] | Tuple[int, int], int]:
    vocab_map: Dict[Tuple[int] | Tuple[int, int], int] = {(i,): i for i in range(256)}
    current_sequence: List[int] = input_ids
    merges_completed: int = 0
    if max_vocab_size:
        max_merges = max(0, max_vocab_size - 256)
    else:
        max_merges = 2000
    while merges_completed < max_merges:
        pair_stats = get_stats(current_sequence)
        if not pair_stats:
            break
        most_frequent_pair: Tuple[int, int] = pair_stats.most_common(1)[0][0]
        freq = pair_stats[most_frequent_pair]
        if freq == 1:
            break
        next_id = len(vocab_map)
        vocab_map[most_frequent_pair] = next_id
        new_sequence = merge(current_sequence, most_frequent_pair, next_id)
        current_sequence = new_sequence
        merges_completed += 1

    print(
        f"Initial sequence length: {len(input_ids)} | Final sequence length: {len(current_sequence)} | Vocab size: {len(vocab_map)} | Compressed ratio: {len(input_ids) / len(current_sequence)}"
    )
    return vocab_map


def get_bytes_from_token_tree(
    token: int, ids_to_pairs: Dict[int, Tuple[int] | Tuple[int, int]]
) -> bytes:
    if token < 256:
        return bytes([token])
    pair = ids_to_pairs[token]
    return b"".join(get_bytes_from_token_tree(b, ids_to_pairs) for b in pair)


def decode_bytes(
    token_ids: List[int], vocab_map: Dict[Tuple[int] | Tuple[int, int], int]
) -> bytes:
    # Create a reverse mapping with the same type as vocab_map
    ids_to_pairs: Dict[int, Tuple[int] | Tuple[int, int]] = {
        v: k for k, v in vocab_map.items()
    }
    text_bytes = b""
    for token in token_ids:
        text_bytes += get_bytes_from_token_tree(token, ids_to_pairs)
    return text_bytes


def decode(
    token_ids: List[int], vocab_map: Dict[Tuple[int] | Tuple[int, int], int]
) -> str:
    text_bytes = decode_bytes(token_ids, vocab_map)
    text = text_bytes.decode("utf-8", errors="replace")
    return text


def encode_bytes(
    text_bytes: bytes, vocab_map: Dict[int | Tuple[int, int | None], int]
) -> List[int]:
    vocab_map_minus_single_bytes = {k: v for k, v in vocab_map.items()}
    ids = list(text_bytes)
    while len(ids) >= 2:
        pairs = get_stats(ids)
        pair = min(
            pairs, key=lambda p: vocab_map_minus_single_bytes.get(p, float("inf"))
        )
        if pair not in vocab_map_minus_single_bytes:
            break
        idx = vocab_map_minus_single_bytes[pair]
        ids = merge(ids, pair, idx)
    return ids


def encode(text: str, vocab_map: Dict[int | Tuple[int, int | None], int]) -> List[int]:
    return encode_bytes(text.encode("utf-8"), vocab_map)
