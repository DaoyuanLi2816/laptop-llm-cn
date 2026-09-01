"""FastAPI Serving：OpenAI 兼容聊天接口、SSE 流式输出与零构建网页。"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Iterator
from typing import Literal

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from laptop_llm.engine import load_inference_bundle
from laptop_llm.generation import GenerationConfig, TokenGenerator


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class ChatCompletionRequest(BaseModel):
    model: str = "laptop-llm"
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    max_tokens: int = Field(default=128, ge=1, le=2048)
    seed: int | None = None


def create_app(
    checkpoint: str,
    *,
    device: str = "auto",
    dtype: str = "auto",
    api_key: str | None = None,
) -> FastAPI:
    model, tokenizer, resolved_device = load_inference_bundle(
        checkpoint, device_name=device, dtype_name=dtype
    )
    generator = TokenGenerator(model, tokenizer, resolved_device)
    # 这是面向单台笔记本的教学 server。串行化 GPU 生成可避免多个请求同时撑爆显存。
    generation_lock = threading.Lock()
    app = FastAPI(
        title="LaptopLLM API",
        version="0.1.0",
        description="本地小模型的 OpenAI 兼容接口",
    )

    def authorize(authorization: str | None) -> None:
        if api_key is None:
            return
        if authorization != f"Bearer {api_key}":
            raise HTTPException(status_code=401, detail="API key 不正确")

    @app.get("/health")
    def health() -> dict[str, str | int]:
        return {
            "status": "ok",
            "device": str(resolved_device),
            "parameters": model.num_parameters(),
            "max_seq_len": model.config.max_seq_len,
        }

    @app.get("/v1/models")
    def models(authorization: str | None = Header(default=None)) -> dict:
        authorize(authorization)
        return {
            "object": "list",
            "data": [
                {
                    "id": "laptop-llm",
                    "object": "model",
                    "created": 0,
                    "owned_by": "local",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(
        request: ChatCompletionRequest,
        authorization: str | None = Header(default=None),
    ):
        authorize(authorization)
        config = GenerationConfig(
            max_new_tokens=min(request.max_tokens, model.config.max_seq_len - 1),
            temperature=request.temperature,
            top_p=request.top_p,
            seed=request.seed,
        )
        messages = [message.model_dump() for message in request.messages]
        prompt_budget = model.config.max_seq_len - config.max_new_tokens
        try:
            prompt_ids = tokenizer.build_chat_prompt(
                messages, add_generation_prompt=True, max_length=max(8, prompt_budget)
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        if request.stream:
            return StreamingResponse(
                stream_completion(
                    generator,
                    generation_lock,
                    prompt_ids,
                    config,
                    completion_id,
                    created,
                    request.model,
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        with generation_lock:
            text, completion_ids = generator.generate_text(prompt_ids, config)
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(completion_ids),
                "total_tokens": len(prompt_ids) + len(completion_ids),
            },
        }

    @app.get("/", response_class=HTMLResponse)
    def web_chat() -> str:
        return CHAT_HTML

    return app


def stream_completion(
    generator: TokenGenerator,
    lock: threading.Lock,
    prompt_ids: list[int],
    config: GenerationConfig,
    completion_id: str,
    created: int,
    model_name: str,
) -> Iterator[str]:
    first = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(first, ensure_ascii=False)}\n\n"
    with lock:
        for text in generator.stream_text(prompt_ids, config):
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [
                    {"index": 0, "delta": {"content": text}, "finish_reason": None}
                ],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
    final = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


CHAT_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
  <title>LaptopLLM 本地聊天</title>
  <style>
    :root{font-family:system-ui,sans-serif;color:#182230;background:#f4f6f8}
    main{max-width:800px;margin:3rem auto;padding:0 1rem}.card{background:white;border-radius:18px;
    box-shadow:0 12px 36px #172b4d18;overflow:hidden}header{padding:1.2rem 1.5rem;background:#172b4d;color:white}
    #chat{height:55vh;overflow:auto;padding:1.2rem}.msg{padding:.75rem 1rem;margin:.6rem 0;border-radius:14px;
    white-space:pre-wrap}.user{background:#e7f0ff;margin-left:18%}.assistant{background:#f0f2f5;margin-right:18%}
    form{display:flex;gap:.7rem;padding:1rem;border-top:1px solid #e7e9ec}textarea{flex:1;resize:none;
    border:1px solid #ccd2d9;border-radius:12px;padding:.8rem;font:inherit}button{border:0;border-radius:12px;
    padding:0 1.2rem;background:#246bfe;color:white;font-weight:650;cursor:pointer}small{opacity:.72}
  </style>
</head>
<body><main><div class="card"><header><b>LaptopLLM</b><br><small>模型与数据都留在你的电脑上</small></header>
<div id="chat"></div><form id="form"><textarea id="input" rows="2" placeholder="输入消息…"></textarea>
<button>发送</button></form></div></main>
<script>
const chat=document.querySelector('#chat'),form=document.querySelector('#form'),input=document.querySelector('#input');
const history=[{role:'system',content:'你是一个诚实、友好、简洁的中文助手。'}];
function add(role,text){const el=document.createElement('div');el.className='msg '+role;el.textContent=text;chat.append(el);chat.scrollTop=chat.scrollHeight;return el}
form.onsubmit=async(e)=>{e.preventDefault();const text=input.value.trim();if(!text)return;input.value='';add('user',text);
history.push({role:'user',content:text});const out=add('assistant','思考中…');
try{const r=await fetch('/v1/chat/completions',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({model:'laptop-llm',messages:history,temperature:.8,max_tokens:160,stream:true})});
if(!r.ok){const data=await r.json();throw new Error(data.detail||r.statusText)}
out.textContent='';let answer='',buffer='';const reader=r.body.getReader(),decoder=new TextDecoder();
while(true){const {value,done}=await reader.read();if(done)break;buffer+=decoder.decode(value,{stream:true});
const events=buffer.split('\n\n');buffer=events.pop();for(const event of events){const line=event.split('\n').find(x=>x.startsWith('data: '));
if(!line||line==='data: [DONE]')continue;const data=JSON.parse(line.slice(6));const delta=data.choices?.[0]?.delta?.content||'';
answer+=delta;out.textContent=answer;chat.scrollTop=chat.scrollHeight}}
history.push({role:'assistant',content:answer});}catch(err){out.textContent='请求失败：'+err.message}}
</script></body></html>"""
