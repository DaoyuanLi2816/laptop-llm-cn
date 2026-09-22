import torch
from fastapi.testclient import TestClient

from laptop_llm.config import ModelConfig
from laptop_llm.model import LaptopLLM
from laptop_llm.server import create_app
from laptop_llm.tokenizer import train_tokenizer


def make_checkpoint(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("你好，这是本地模型。\n请解释语言模型。\n" * 6, encoding="utf-8")
    tokenizer = train_tokenizer(
        [corpus], tmp_path / "tokenizer.json", vocab_size=320, min_frequency=1
    )
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        dim=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=32,
    )
    model = LaptopLLM(config)
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "format_version": 1,
            "stage": "sft",
            "step": 1,
            "model_config": config.to_dict(),
            "model": model.state_dict(),
            "tokenizer_json": tokenizer.to_str(),
        },
        checkpoint,
    )
    return checkpoint


def test_health_and_openai_compatible_completion(tmp_path):
    app = create_app(str(make_checkpoint(tmp_path)), device="cpu", dtype="float32")
    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "laptop-llm",
            "messages": [{"role": "user", "content": "你好"}],
            "temperature": 0,
            "max_tokens": 2,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["message"]["role"] == "assistant"
    assert payload["usage"]["completion_tokens"] <= 2


def test_api_key_is_enforced(tmp_path):
    app = create_app(
        str(make_checkpoint(tmp_path)), device="cpu", dtype="float32", api_key="secret"
    )
    client = TestClient(app)
    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers={"Authorization": "Bearer secret"}).status_code == 200


def test_streaming_completion_ends_with_done(tmp_path):
    app = create_app(str(make_checkpoint(tmp_path)), device="cpu", dtype="float32")
    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "你好"}],
            "stream": True,
            "temperature": 0,
            "max_tokens": 2,
        },
    )
    assert response.status_code == 200
    assert "chat.completion.chunk" in response.text
    assert response.text.rstrip().endswith("data: [DONE]")


def test_context_overflow_is_explicit_and_ui_script_is_not_broken(tmp_path):
    import re

    app = create_app(str(make_checkpoint(tmp_path)), device="cpu", dtype="float32")
    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "x" * 500}]}
    )
    assert response.status_code == 400
    html = client.get("/").text
    assert "split('\\n\\n')" in html
    assert not re.search(r"split\('\n", html)
    assert 'id="reset"' in html and 'id="apiKey"' in html
