import json

import torch

from laptop_llm.data import (
    PackedTokenDataset,
    PreferenceDataset,
    SFTDataset,
    pad_lm_batch,
    pad_preference_batch,
    prepare_pretrain_cache,
)
from laptop_llm.tokenizer import LLMTokenizer, train_tokenizer


def make_tokenizer(tmp_path) -> LLMTokenizer:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "你好，世界。\n语言模型学习下一个 token。\n任意 emoji 也可以编码：🌱\n" * 4,
        encoding="utf-8",
    )
    return train_tokenizer([corpus], tmp_path / "tokenizer.json", vocab_size=320, min_frequency=1)


def test_byte_bpe_roundtrip_and_special_tokens(tmp_path):
    tokenizer = make_tokenizer(tmp_path)
    text = "中文 + English + 🌱"
    ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    assert ids[0] == tokenizer.bos_id
    assert ids[-1] == tokenizer.eos_id
    assert tokenizer.decode(ids) == text


def test_sft_masks_user_and_supervises_assistant(tmp_path):
    tokenizer = make_tokenizer(tmp_path)
    messages = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好，很高兴见到你。"},
    ]
    input_ids, labels = tokenizer.build_sft_example(messages, max_length=64)
    assistant_position = input_ids.index(tokenizer.assistant_id)
    assert all(label == -100 for label in labels[: assistant_position + 1])
    assert any(label != -100 for label in labels[assistant_position + 1 :])

    path = tmp_path / "sft.jsonl"
    path.write_text(json.dumps({"messages": messages}, ensure_ascii=False) + "\n", encoding="utf-8")
    dataset = SFTDataset(path, tokenizer, max_length=64)
    batch = pad_lm_batch([dataset[0], dataset[0]], tokenizer.pad_id)
    assert batch["input_ids"].shape == batch["labels"].shape
    assert batch["attention_mask"].dtype == torch.bool


def test_packed_cache_has_exact_next_token_targets(tmp_path):
    tokenizer = make_tokenizer(tmp_path)
    source = tmp_path / "pretrain.txt"
    source.write_text("这是第一段文本。\n这是第二段更长的文本。\n" * 8, encoding="utf-8")
    cache = tmp_path / "tokens.bin"
    metadata = prepare_pretrain_cache(source, cache, tokenizer)
    assert metadata["documents"] == 16
    dataset = PackedTokenDataset(cache, block_size=8)
    sample = dataset[0]
    assert torch.equal(sample["input_ids"][1:], sample["next_token_labels"][:-1])


def test_preference_dataset_and_collator(tmp_path):
    tokenizer = make_tokenizer(tmp_path)
    row = {
        "prompt": [{"role": "user", "content": "一加一是多少？"}],
        "chosen": "一加一等于二。",
        "rejected": "一加一等于三。",
    }
    path = tmp_path / "dpo.jsonl"
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    dataset = PreferenceDataset(path, tokenizer, max_length=64)
    batch = pad_preference_batch([dataset[0]], tokenizer.pad_id)
    assert set(batch) == {"chosen", "rejected"}
    assert (batch["chosen"]["labels"] != -100).any()
