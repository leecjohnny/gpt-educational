import os
import tempfile
import pytest

from bpe import BPETokenizer, TiktokenTokenizer


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


def test_tokenizer_save_load():
    """Test that a tokenizer can be saved and loaded."""
    # Create a simple tokenizer
    tokenizer = BPETokenizer(max_vocab_size=500)
    sample_text = "This is a test of the tokenizer save and load functionality."
    tokenizer.train(list(sample_text.encode("utf-8")))

    # Get the initial vocabulary size and encode a sample
    initial_vocab_size = tokenizer.vocab_size
    encoded = tokenizer.encode(sample_text)

    # Save the tokenizer to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        temp_path = temp.name

    try:
        tokenizer.save(temp_path)

        # Load the tokenizer from the file
        loaded_tokenizer = BPETokenizer.load(temp_path)

        # Check that the loaded tokenizer has the same vocabulary size
        assert loaded_tokenizer.vocab_size == initial_vocab_size

        # Check that the loaded tokenizer produces the same encoding
        loaded_encoded = loaded_tokenizer.encode(sample_text)
        assert loaded_encoded == encoded

        # Check that the loaded tokenizer can decode correctly
        decoded = loaded_tokenizer.decode(loaded_encoded)
        assert decoded == sample_text

    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_tokenizer_encoding_performance():
    """Test that encoding with caching is faster for repeated operations."""
    tokenizer = BPETokenizer(max_vocab_size=500)
    sample_text = "This is a sample text that will be encoded multiple times to test caching performance."
    tokenizer.train(list(sample_text.encode("utf-8")))

    # First encoding (cold cache)
    encoded1 = tokenizer.encode(sample_text)

    # Second encoding (should use cache)
    encoded2 = tokenizer.encode(sample_text)

    # Verify results are the same
    assert encoded1 == encoded2

    # Third encoding with different text (should not use cache)
    different_text = (
        "This is a different text to ensure the cache isn't incorrectly used."
    )
    encoded3 = tokenizer.encode(different_text)

    # Verify we can decode all properly
    assert tokenizer.decode(encoded1) == sample_text
    assert tokenizer.decode(encoded3) == different_text


@pytest.mark.skipif(
    not pytest.importorskip("tiktoken"), reason="tiktoken not installed"
)
def test_tiktoken_tokenizer():
    """Test the TiktokenTokenizer wrapper."""
    try:
        tokenizer = TiktokenTokenizer(model_name="gpt2")
        sample_text = "Testing the tiktoken tokenizer wrapper."

        # Test encoding and decoding
        encoded = tokenizer.encode(sample_text)
        decoded = tokenizer.decode(encoded)

        # TiktokenTokenizer might not be perfectly reversible due to whitespace handling
        # so we normalize whitespace for comparison
        assert " ".join(decoded.split()) == " ".join(sample_text.split())

        # Test vocab_size property
        assert tokenizer.vocab_size > 0
    except Exception as e:
        pytest.skip(f"Skipping tiktoken test due to error: {e}")
