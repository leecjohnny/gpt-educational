from collections import Counter
from typing import Dict, List, Tuple, Union


def get_stats(token_ids: List[int]) -> Dict[Tuple[int, int], int]:
    """
    Given a list of integers, return a dictionary of counts of consecutive pairs
    Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    """
    int_pairs = zip(token_ids, token_ids[1:])
    int_pair_counts = Counter(int_pairs)
    return int_pair_counts


def merge(existing_ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
    """
    In the list of integers (existing_ids), replace all consecutive occurrences
    of pair with the new integer token new_id
    Example: existing_ids=[1, 2, 3, 1, 2], pair=(1, 2), new_id=4 -> [4, 3, 4]
    """
    new_ids = []
    assert len(existing_ids) > 1, "Not enough pairs to merge"
    idx_to_skip: int = None
    for idx, (i0, i1) in enumerate(zip(existing_ids, existing_ids[1:] + [None])):
        if idx == idx_to_skip:
            continue
        elif i0 == pair[0] and i1 == pair[1]:
            new_ids.append(new_id)
            idx_to_skip = idx + 1
            continue
        new_ids.append(i0)
    return new_ids


def test_merge() -> None:
    existing_ids = [1, 2, 3, 1, 2, 9, 10, 7, 1, 2, 3]
    pair = (1, 2)
    new_id = 4
    result = merge(existing_ids, pair, new_id)
    assert result == [
        4,
        3,
        4,
        9,
        10,
        7,
        4,
        3,
    ], f"Expected [4, 3, 4, 9, 10, 7, 4, 3], but got {result}"


def train_bpe_vocab(
    input_ids: List[int], max_vocab_size: int | None = None
) -> Dict[Union[int, Tuple[int, int]], int]:
    vocab_map: Dict[Tuple[int, int | None], int] = {
        tuple([i, None]): i for i in range(256)
    }
    current_sequence: List[int] = input_ids
    merges_completed: int = 0
    if max_vocab_size:
        max_merges = max(0, max_vocab_size - 256)
    else:
        max_merges = 2000
    while merges_completed < max_merges:
        pair_stats = get_stats(current_sequence)
        (most_frequent_pair, freq) = pair_stats.most_common(1)[0]
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
    token: int, vocab_map: Dict[Union[int, Tuple[int, int]], int]
) -> bytes:
    if token < 256:
        return bytes([token])
    return b"".join(get_bytes_from_token_tree(b, vocab_map) for b in vocab_map[token])


def decode_bytes(
    token_ids: List[int], vocab_map: Dict[Union[int, Tuple[int, int]], int]
) -> bytes:
    ids_to_pairs = {value: key for key, value in vocab_map.items()}
    text_bytes = b""
    for token in token_ids:
        text_bytes += get_bytes_from_token_tree(token, ids_to_pairs)
    return text_bytes


def decode(
    token_ids: List[int], vocab_map: Dict[Union[int, Tuple[int, int]], int]
) -> str:
    text_bytes = decode_bytes(token_ids, vocab_map)
    text = text_bytes.decode("utf-8", errors="replace")
    return text


def encode_bytes(
    text_bytes: bytes, vocab_map: Dict[Union[int, Tuple[int, int]], int]
) -> List[int]:
    vocab_map_minus_single_bytes = {k: v for k, v in vocab_map.items() if len(k) > 1}
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


def encode(text: str, vocab_map: Dict[Union[int, Tuple[int, int]], int]) -> List[int]:
    return encode_bytes(text.encode("utf-8"), vocab_map)


def test_encode_decode():
    text = """hello world!!!!!From North America to other continents
    text_bytes) # list of integers in range 0..255

        # iteratively merge the most common pairs to create new tokens
        for i in range(num_merges):
            # count up the number of times every consecutive pair appears
            stats = get_stats(ids)
            # find the pair with the highest count
            pair = max(stats, key=stats.get)
            # mint a new token: assign it the next available id
            idx = 256 + i
            # replace all occurrences of pair in ids with idx
            ids = merge(ids, pair, idx)
            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[p
From Europe to other continents"""
    vocab_map = train_bpe_vocab(list(text.encode("utf-8")))
    print(vocab_map)
    encoded = encode(text, vocab_map)
    print(f"{text} -> {encoded}")
    decoded = decode(encoded, vocab_map)
    assert text == decoded, f"'{text}' != '{decoded}'"
