import pytest

from bpe import BPETokenizer


@pytest.mark.parametrize(
    "existing_ids,pair,new_id,expected",
    [
        (
            [1, 2, 3, 1, 2, 9, 10, 7, 1, 2, 3],
            (1, 2),
            4,
            [4, 3, 4, 9, 10, 7, 4, 3],
        )
    ],
)
def test_merge(existing_ids, pair, new_id, expected):
    tokenizer = BPETokenizer()
    result = tokenizer._merge(existing_ids, pair, new_id)
    assert result == expected, f"Expected {expected}, but got {result}"


@pytest.mark.parametrize(
    "text",
    [
        """hello world!!!!!From North America to other continents
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
    ],
)
def test_encode_decode(text):
    tokenizer = BPETokenizer()
    tokenizer.train(list(text.encode("utf-8")))
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    assert text == decoded, f"'{text}' != '{decoded}'"
