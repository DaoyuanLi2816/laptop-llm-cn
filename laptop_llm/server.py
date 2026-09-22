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
        version="0.2.0",
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
        prompt_budget = model.config.max_seq_len - 1
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
            finish_reason = generator.finish_reason
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": finish_reason,
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
                "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        finish_reason = generator.finish_reason
    final = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
    }
    yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


CHAT_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
  <title>LaptopLLM · 本地研究实验室</title>
  <style>
    :root{font-family:system-ui,sans-serif;color:#182230;background:#eff3f1}
    main{max-width:1000px;margin:3rem auto;padding:0 1rem}.card{background:white;border-radius:18px;
    box-shadow:0 12px 36px #172b4d18;overflow:hidden}header{padding:1.2rem 1.5rem;background:#172b4d;color:white}
    #chat{height:45vh;overflow:auto;padding:1.2rem}.msg{padding:.75rem 1rem;margin:.6rem 0;border-radius:14px;
    white-space:pre-wrap}.user{background:#e7f0ff;margin-left:18%}.assistant{background:#f0f2f5;margin-right:18%}
    form{display:flex;gap:.7rem;padding:1rem;border-top:1px solid #e7e9ec}textarea{flex:1;resize:none;
    border:1px solid #ccd2d9;border-radius:12px;padding:.8rem;font:inherit}button{border:0;border-radius:12px;
    padding:.7rem 1.2rem;background:#19715b;color:white;font-weight:650;cursor:pointer}small{opacity:.82}
    .eyebrow{font-size:.75rem;letter-spacing:.15em;color:#19715b}h1{font-size:2.6rem;letter-spacing:-.04em;margin:.5rem 0}
    .intro{color:#52665e;line-height:1.7;margin-bottom:1.6rem}.settings{padding:1rem;display:flex;gap:1rem;flex-wrap:wrap;background:#f8faf9}
    label{font-size:.8rem;display:flex;flex-direction:column;gap:.3rem}input,select{padding:.5rem;border:1px solid #ccd8d1;border-radius:6px;max-width:160px}
    #status{font-size:.8rem;color:#455c50;padding:.7rem 1rem}button:disabled{opacity:.4;cursor:wait}footer{font-size:.8rem;color:#5b6c64;line-height:1.7;margin:1rem 0}
    .empty{color:#74837c;text-align:center;padding:3rem 0}header{background:#173e32;display:flex;justify-content:space-between;align-items:center}
    @media(max-width:600px){main{margin:1rem auto}h1{font-size:2rem}.user{margin-left:5%}.assistant{margin-right:5%}}
  </style>
</head>
<body><main><div class="eyebrow">LAPTOP LLM / RESEARCH LAB</div><h1>把大模型，拆开学。</h1>
<p class="intro">从 token 到推理，从奖励到策略更新。这里运行的是你自己的本地 checkpoint。<br>小规模验证算法，不把流水线跑通当作智能证明。</p>
<div class="card"><header><div><b>本地试聊</b><br><small>无云端 API · 无需付费算力</small></div><button id="reset" type="button">新对话</button></header>
<div class="settings"><label>采样温度<input id="temperature" type="number" min="0" max="2" step="0.1" value="0.8"></label>
<label>最多生成 token<input id="maxTokens" type="number" min="1" max="2048" value="64"></label>
<label>API key（如启用）<input id="apiKey" type="password" autocomplete="off" placeholder="仅内存使用"></label>
<label>学习提示<select id="example"><option value="">选择一个问题</option><option>你好</option><option>解释 KV Cache 的作用。</option><option>2+3=? 请用 &lt;answer&gt;标签回答。</option></select></label></div>
<div id="status" role="status">正在读取本机模型信息…</div><div id="chat" aria-live="polite"><div class="empty">从一句“你好”开始。<br>smoke 权重通常只会输出随机文本。</div></div>
<form id="form"><textarea id="input" aria-label="消息" rows="2" placeholder="输入消息；超长时请新建对话或缩短问题…"></textarea>
<button id="send">发送</button></form></div><footer>若模型生成 &lt;think&gt; 内容，它只是可见推理文本，不代表可验证的内部思维。<br>仅绑定 localhost；该串行教学服务没有生产级队列、取消调度与安全保障。</footer></main>
<script>
const chat=document.querySelector('#chat'),form=document.querySelector('#form'),input=document.querySelector('#input');
let history=[],busy=false;
const send=document.querySelector('#send'),reset=document.querySelector('#reset'),status=document.querySelector('#status');
fetch('/health').then(r=>r.json()).then(d=>{status.textContent=`设备 ${d.device} · ${(d.parameters/1e6).toFixed(2)}M 参数 · 上下文 ${d.max_seq_len} tokens`}).catch(()=>{status.textContent='无法连接本地模型'});
reset.onclick=()=>{if(busy)return;history=[];chat.replaceChildren()};
document.querySelector('#example').onchange=e=>{input.value=e.target.value};
function add(role,text){const el=document.createElement('div');el.className='msg '+role;el.textContent=text;chat.append(el);chat.scrollTop=chat.scrollHeight;return el}
form.onsubmit=async(e)=>{e.preventDefault();const text=input.value.trim();if(!text||busy)return;
busy=true;send.disabled=true;reset.disabled=true;chat.querySelector('.empty')?.remove();input.value='';add('user',text);
const candidate=[...history,{role:'user',content:text}],out=add('assistant','生成中…');
const headers={'Content-Type':'application/json'},key=document.querySelector('#apiKey').value;if(key)headers.Authorization='Bearer '+key;
try{const r=await fetch('/v1/chat/completions',{method:'POST',headers,
body:JSON.stringify({model:'laptop-llm',messages:candidate,temperature:Number(document.querySelector('#temperature').value),max_tokens:Number(document.querySelector('#maxTokens').value),stream:true})});
if(!r.ok){const data=await r.json();throw new Error(data.detail||r.statusText)}
out.textContent='';let answer='',buffer='';const reader=r.body.getReader(),decoder=new TextDecoder();
while(true){const {value,done}=await reader.read();if(done)break;buffer+=decoder.decode(value,{stream:true});
const events=buffer.split('\n\n');buffer=events.pop();for(const event of events){const line=event.split('\n').find(x=>x.startsWith('data: '));
if(!line||line==='data: [DONE]')continue;const data=JSON.parse(line.slice(6));const delta=data.choices?.[0]?.delta?.content||'';
answer+=delta;out.textContent=answer;chat.scrollTop=chat.scrollHeight}}
if(answer.trim()){history=[...candidate,{role:'assistant',content:answer}]}else{out.textContent='模型没有生成可见文本；历史未追加空回复。'}
}catch(err){out.textContent='请求失败：'+err.message}finally{busy=false;send.disabled=false;reset.disabled=false;input.focus()}}
</script></body></html>"""
