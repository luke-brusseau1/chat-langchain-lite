"""Chat LangChain Lite — FastHTML frontend.

Mounted into the LangGraph server as a custom app via ``langgraph.json``'s
``http.app`` key (``"./web/app.py:app"``), so it is served from the same origin
as the graph API. It renders a dark chat UI styled after chat.langchain.com and
streams the agent's response token-by-token over SSE.

Features:
  * Streaming chat with client-side markdown rendering (marked.js + highlight.js).
  * Per-response user feedback (👍/👎 + optional comment) written to LangSmith.
    The browser POSTs to the same-origin ``/feedback`` endpoint, which submits
    via the server-side LangSmith client, so the API key never reaches the browser.
  * A "Trace" deep-link per response, resolved server-side via ``/trace/{run_id}``.
  * A collapsible sidebar of prior conversations (thread list in localStorage;
    history rendered from the graph's own thread state).

The frontend never imports the graph directly — it reaches it over the LangGraph
SDK (`langgraph_sdk`) on localhost: port 2024 under `langgraph dev`, port 8000 in
the deployed container. The base URL comes from ``LANGGRAPH_API_URL`` and is
constrained to loopback (see ``_api_url``).
"""

import asyncio
import base64
import contextlib
import json
import os
import re
import secrets
import uuid
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from fasthtml.common import (
    FT,
    H1,
    A,
    Aside,
    Button,
    Div,
    EventStream,
    Form,
    Header,
    Img,
    Input,
    Link,
    Main,
    Nav,
    P,
    Script,
    Span,
    Style,
    fast_app,
    sse_message,
)
from langgraph_sdk import get_client
from langsmith import Client
from langsmith.schemas import FeedbackConfig
from starlette.responses import PlainTextResponse, RedirectResponse

from context import CONTEXT_HUB_REPO

load_dotenv(override=True)

ASSISTANT_ID = "chat_langchain_lite"

# The application slug — matches the agent's own run naming (agent/agent.py's
# `_config`) so the chat UI's traces line up with the scripted path.
APP_SLUG = "chat-lc-lite"

# Human-feedback keys emitted by this chat UI and consumed by the monitoring /
# online-eval automation. Keep these names stable so the two can't drift.
SCORE_KEY = "user_score"
COMMENT_KEY = "user_comment"

# Loopback hosts the frontend is allowed to reach. The frontend and the graph run
# in the same process/container, so the SDK target is always localhost —
# constraining to loopback closes the SSRF surface flagged in review.
_ALLOWED_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]"}
# Only ever redirect the trace link to LangSmith (open-redirect / SSRF guard).
_TRACE_HOST_SUFFIX = ("smith.langchain.com",)
_UUID_RE = re.compile(
    r"\A[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\Z"
)


def _api_url() -> str:
    """Resolve the LangGraph API base URL (loopback-only).

    Explicit ``LANGGRAPH_API_URL`` wins; otherwise default to the
    ``langgraph dev`` port (2024). In a deployment, set
    ``LANGGRAPH_API_URL=http://localhost:8000`` (the container's API port).
    """
    url = os.getenv("LANGGRAPH_API_URL", "http://localhost:2024")
    host = urlparse(url).hostname or ""
    if host not in _ALLOWED_HOSTS:
        raise ValueError(
            f"LANGGRAPH_API_URL host {host!r} is not loopback; "
            "the frontend may only reach the co-located graph server."
        )
    return url


_LS_CLIENT: Client | None = None


def _ls() -> Client:
    """Lazily-built LangSmith client (uses LANGSMITH_API_KEY from env)."""
    global _LS_CLIENT
    if _LS_CLIENT is None:
        _LS_CLIENT = Client()
    return _LS_CLIENT


def _logo_data_uri() -> str:
    path = Path(__file__).resolve().parent / "langchain-color.png"
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/png;base64,{b64}"


LOGO = _logo_data_uri()

SUGGESTIONS = [
    "Walk me through building a LangGraph agent with middleware, persistence, and streaming — include code.",
    "Show me how to set up LangSmith tracing and offline evals end-to-end.",
    "What is LangSmith and what is it used for?",
    "Help me debug my Django view function — it's throwing a 500.",
    "Where can I find the official LangChain documentation?",
    "What's the minimum Python version for LangGraph?",
]

# ── Styling (dark, sidebar + centered column — inspired by chat.langchain.com) ──
CSS = """
:root{
  --bg:#0a0a0b; --panel:#0f0f11; --panel-2:#141417; --border:#222226;
  --text:#ececee; --muted:#8a8a93; --faint:#5b5b63;
  --accent:#3b82f6; --accent-2:#60a5fa; --code:#93c5fd;
  --user:#161619; --sidebar:#0c0c0e; --radius:12px;
}
*{box-sizing:border-box;}
html,body{margin:0;height:100%;}
body{background:var(--bg);color:var(--text);
  font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif;
  font-size:14px;line-height:1.6;-webkit-font-smoothing:antialiased;}
a{color:inherit;text-decoration:none;}
button{font-family:inherit;cursor:pointer;}

.app{display:flex;height:100vh;overflow:hidden;}

/* ── Sidebar ── */
.sidebar{width:264px;flex:0 0 264px;background:var(--sidebar);
  border-right:1px solid var(--border);display:flex;flex-direction:column;
  transition:margin-left .22s ease,opacity .2s ease;}
.app.collapsed .sidebar{margin-left:-264px;opacity:0;}
.sb-head{display:flex;align-items:center;gap:8px;padding:14px 14px 10px;}
.sb-brand{display:flex;align-items:center;gap:8px;font-weight:600;font-size:14px;flex:1;}
.sb-brand img{width:22px;height:22px;}
.icon-btn{background:transparent;border:1px solid transparent;color:var(--muted);
  width:32px;height:32px;border-radius:8px;display:flex;align-items:center;
  justify-content:center;font-size:16px;transition:background .15s,color .15s;}
.icon-btn:hover{background:var(--panel-2);color:var(--text);}
.sb-new{margin:4px 12px 12px;padding:9px 12px;border:1px solid var(--border);
  border-radius:10px;color:var(--text);font-size:13px;font-weight:500;
  display:flex;align-items:center;gap:8px;transition:border-color .15s,background .15s;}
.sb-new:hover{border-color:var(--accent);background:var(--panel-2);}
.sb-section{padding:6px 18px;font-size:11px;font-weight:600;letter-spacing:.5px;
  text-transform:uppercase;color:var(--faint);}
.sb-threads{flex:1;overflow-y:auto;padding:0 8px 12px;}
.sb-thread{display:block;padding:9px 12px;border-radius:8px;color:var(--muted);
  font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
  transition:background .12s,color .12s;}
.sb-thread:hover{background:var(--panel-2);color:var(--text);}
.sb-thread.active{background:var(--panel-2);color:var(--text);}
.sb-empty{padding:10px 18px;color:var(--faint);font-size:12px;}
.sb-foot{padding:12px 18px;border-top:1px solid var(--border);color:var(--faint);font-size:11px;}

/* ── Main ── */
.main{flex:1;display:flex;flex-direction:column;min-width:0;}
.topbar{display:flex;align-items:center;gap:10px;padding:12px 18px;
  border-bottom:1px solid var(--border);}
/* Brand lives in the sidebar; only surface it in the topbar when collapsed. */
.topbar .brand{display:none;align-items:center;gap:9px;font-weight:600;}
.app.collapsed .topbar .brand{display:flex;}
.topbar .brand img{width:24px;height:24px;}

.chat{flex:1;overflow-y:auto;}
.chat-inner{max-width:768px;margin:0 auto;padding:26px 20px 40px;}

/* Empty state */
.empty{text-align:center;padding:56px 0 26px;}
.empty img{width:60px;height:60px;opacity:.95;}
.empty h1{font-size:24px;font-weight:700;margin:18px 0 6px;letter-spacing:-.4px;}
.empty p{color:var(--muted);margin:0;}
.sug-label{font-size:11px;font-weight:600;letter-spacing:.6px;text-transform:uppercase;
  color:var(--faint);text-align:center;margin:30px 0 14px;}
.sug-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;}
.sug{background:var(--panel);border:1px solid var(--border);color:var(--text);
  border-radius:var(--radius);font-size:13px;line-height:1.5;padding:15px 16px;
  min-height:78px;text-align:left;font-weight:450;
  transition:border-color .15s,transform .15s,background .15s;}
.sug:hover{border-color:var(--accent);background:var(--panel-2);transform:translateY(-1px);}
@media(max-width:620px){.sug-grid{grid-template-columns:1fr;}}

/* Messages */
.msg{display:flex;gap:13px;margin-bottom:22px;}
.avatar{flex:0 0 30px;height:30px;border-radius:8px;display:flex;align-items:center;
  justify-content:center;font-size:15px;}
.msg.user .avatar{background:var(--user);}
.msg.assistant .avatar{background:#15233f;color:var(--code);}
.msg-main{flex:1;min-width:0;}
.bubble{font-size:14.5px;line-height:1.7;overflow-wrap:anywhere;}
.msg.user .bubble{background:var(--user);border:1px solid var(--border);
  border-radius:var(--radius);padding:10px 14px;display:inline-block;}
.bubble p{margin:.5em 0;}.bubble p:first-child{margin-top:0;}.bubble p:last-child{margin-bottom:0;}
.bubble strong{color:#fff;}.bubble h2,.bubble h3,.bubble h4{color:#fff;margin:.8em 0 .4em;}
.bubble a{color:var(--accent-2);text-decoration:underline;}
.bubble ul,.bubble ol{padding-left:1.25em;margin:.5em 0;}
.bubble code:not(pre code){background:#000;color:var(--code);border-radius:5px;
  padding:1px 6px;font-size:12.5px;font-family:'JetBrains Mono',ui-monospace,monospace;}
.bubble pre{background:#000;border:1px solid var(--border);border-radius:10px;
  padding:13px 14px;overflow-x:auto;margin:.6em 0;}
.bubble pre code{background:transparent;color:var(--text);padding:0;font-size:12.5px;}
.cursor{display:inline-block;width:8px;height:15px;background:var(--accent);
  border-radius:1px;animation:blink 1s steps(2) infinite;vertical-align:middle;}
@keyframes blink{0%,50%{opacity:1;}50.01%,100%{opacity:0;}}

/* Feedback / action bar */
.fb{display:flex;align-items:center;gap:6px;margin-top:10px;
  opacity:.55;transition:opacity .15s;flex-wrap:wrap;}
.msg:hover .fb{opacity:1;}
.fb-btn,.fb-cmt-toggle{background:transparent;border:1px solid transparent;
  color:var(--muted);border-radius:7px;padding:4px 8px;font-size:13px;line-height:1;
  transition:background .12s,color .12s,border-color .12s;}
.fb-btn:hover,.fb-cmt-toggle:hover{background:var(--panel-2);color:var(--text);}
/* Selected vote: filled accent, stays bright even while disabled. */
.fb-btn.sel{color:#fff;border-color:var(--accent);background:var(--accent);}
.fb-btn:disabled{cursor:default;}
.fb-btn.sel:disabled{opacity:1;}
.fb:has(.fb-btn.sel) .fb-btn:not(.sel){opacity:.4;}
.fb-trace{color:var(--muted);font-size:12px;padding:4px 8px;border-radius:7px;
  margin-left:auto;transition:color .12s,background .12s;}
.fb-trace:hover{color:var(--accent-2);background:var(--panel-2);}
.fb-cmt{display:flex;gap:6px;width:100%;margin-top:8px;}
.fb-cmt[hidden]{display:none;}
.fb-cmt-input{flex:1;background:var(--panel);border:1px solid var(--border);
  border-radius:8px;color:var(--text);font-size:13px;padding:8px 11px;outline:none;}
.fb-cmt-input:focus{border-color:var(--accent);}
.fb-cmt-send{background:var(--accent);border:none;color:#fff;border-radius:8px;
  padding:8px 14px;font-size:13px;font-weight:500;}
.fb-cmt-send:disabled{opacity:.6;cursor:default;}
.fb-thanks{color:var(--faint);font-size:12px;}

/* Composer */
.composer{border-top:1px solid var(--border);background:linear-gradient(180deg,transparent,var(--bg) 40%);}
.composer-inner{max-width:768px;margin:0 auto;padding:14px 20px 20px;}
.chat-form{display:flex;gap:10px;align-items:flex-end;background:var(--panel);
  border:1px solid var(--border);border-radius:14px;padding:8px 8px 8px 16px;
  transition:border-color .15s;}
.chat-form:focus-within{border-color:var(--accent);}
.chat-input{flex:1;background:transparent;border:none;color:var(--text);
  font-size:14.5px;padding:8px 0;outline:none;resize:none;max-height:160px;}
.chat-input::placeholder{color:var(--faint);}
.send-btn{background:var(--accent);border:none;color:#fff;width:38px;height:38px;
  border-radius:10px;font-size:16px;display:flex;align-items:center;justify-content:center;
  flex:0 0 38px;transition:background .15s,opacity .15s;}
.send-btn:hover{background:var(--accent-2);}
.composer-hint{text-align:center;color:var(--faint);font-size:11px;margin-top:8px;}
"""

HEAD = (
    Link(rel="preconnect", href="https://fonts.googleapis.com"),
    Link(rel="preconnect", href="https://fonts.gstatic.com", crossorigin=""),
    Link(
        rel="stylesheet",
        href="https://fonts.googleapis.com/css2?family=Inter:wght@400;450;500;600;700&family=JetBrains+Mono&display=swap",
    ),
    Link(
        rel="stylesheet",
        href="https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.9.0/build/styles/github-dark.min.css",
    ),
    Script(src="https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.9.0/build/highlight.min.js"),
    Script(src="https://cdn.jsdelivr.net/npm/marked@12.0.0/marked.min.js"),
    Script(src="https://cdn.jsdelivr.net/npm/dompurify@3.0.9/dist/purify.min.js"),
    Script(src="https://cdn.jsdelivr.net/npm/htmx-ext-sse@2.2.3/dist/sse.js"),
    Style(CSS),
)

CLIENT_JS = """
// ---- markdown rendering (MutationObserver: robust against htmx/SSE swap timing) ----
function renderMd(el){
  if(!el||!el.classList||!el.classList.contains('md-live'))return;
  var raw=el.textContent||'';
  if(el.dataset.src===raw) return;            // already rendered this exact text
  el.dataset.src=raw;
  // Content is LLM/agent output (live or from thread history) — untrusted. marked
  // does NOT sanitize, so always run the parsed HTML through DOMPurify before it
  // touches innerHTML, closing the DOM-XSS hole in this single render chokepoint.
  if(window.marked){
    var html=window.marked.parse(raw);
    el.innerHTML=window.DOMPurify?window.DOMPurify.sanitize(html):'';
  }
  if(window.hljs) el.querySelectorAll('pre code').forEach(b=>window.hljs.highlightElement(b));
  el.dataset.src=el.textContent;              // post-render text → dedupe my own mutation
}
function renderAll(){ document.querySelectorAll('.md-live').forEach(renderMd); }
function scrollDown(){ var c=document.getElementById('messages'); if(c) c.scrollTop=c.scrollHeight; }
var _mdObs=new MutationObserver(function(muts){
  var set=new Set();
  muts.forEach(function(m){
    var n=(m.target&&m.target.nodeType===1)?m.target:(m.target&&m.target.parentElement);
    var el=n&&n.closest?n.closest('.md-live'):null;
    if(el) set.add(el);
  });
  set.forEach(renderMd); scrollDown();
});
function startObserver(){ var c=document.getElementById('messages'); if(c) _mdObs.observe(c,{subtree:true,childList:true,characterData:true}); }

// ---- sidebar (collapse state + thread list in localStorage) ----
var TKEY='clc_threads', SBKEY='clc_sb_collapsed';
function loadThreads(){ try{return JSON.parse(localStorage.getItem(TKEY))||[]}catch(e){return[]} }
function saveThreads(t){ localStorage.setItem(TKEY,JSON.stringify(t)); }
function renderThreads(){
  var el=document.getElementById('sb-threads'); if(!el) return;
  var list=loadThreads(), cur=window.CLC_THREAD;
  if(!list.length){ el.innerHTML='<div class="sb-empty">No conversations yet</div>'; return; }
  el.innerHTML='';
  list.slice().sort((a,b)=>b.ts-a.ts).forEach(function(t){
    var a=document.createElement('a');
    a.className='sb-thread'+(t.id===cur?' active':'');
    a.href='/?thread='+encodeURIComponent(t.id);
    a.textContent=t.title||'New conversation';
    a.title=a.textContent;
    el.appendChild(a);
  });
}
function upsertThread(title){
  var id=window.CLC_THREAD; if(!id) return;
  var list=loadThreads(), e=list.find(x=>x.id===id);
  if(!e){ list.push({id:id,title:(title||'').slice(0,80),ts:Date.now()}); }
  else { if(!e.title&&title) e.title=title.slice(0,80); e.ts=Date.now(); }
  saveThreads(list); renderThreads();
}
function toggleSidebar(){
  var app=document.getElementById('app'); app.classList.toggle('collapsed');
  localStorage.setItem(SBKEY, app.classList.contains('collapsed')?'1':'0');
}
function applySidebar(){
  if(localStorage.getItem(SBKEY)==='1') document.getElementById('app').classList.add('collapsed');
}

// ---- feedback (same-origin POST -> server upserts via stable feedback_id) ----
function postFeedback(body){
  return fetch('/feedback',{method:'POST',
    headers:{'Content-Type':'application/x-www-form-urlencoded'},
    body:new URLSearchParams(body)});
}
function vote(btn){
  var fb=btn.closest('.fb'), prev=fb.querySelector('.fb-btn.sel');
  // Each click supersedes the previous. Stamp this request so a stale, late-
  // rejecting fetch from an earlier click can't revert a newer selection.
  var seq=(fb._voteSeq=(fb._voteSeq||0)+1);
  // Select clicked (highlight + disable); re-enable the other so you can switch.
  fb.querySelectorAll('.fb-btn').forEach(b=>{ b.classList.remove('sel'); b.disabled=false; });
  btn.classList.add('sel'); btn.disabled=true;
  postFeedback({run_id:btn.dataset.run,kind:'score',value:btn.dataset.score})
    .then(r=>{ if(!r.ok) throw 0; })
    .catch(()=>{                       // revert to prior selection on failure
      if(fb._voteSeq!==seq) return;    // superseded by a newer vote — leave it
      btn.classList.remove('sel'); btn.disabled=false;
      if(prev){ prev.classList.add('sel'); prev.disabled=true; }
    });
}
function sendComment(btn){
  var fb=btn.closest('.fb'), input=fb.querySelector('.fb-cmt-input'), text=(input.value||'').trim();
  if(!text) return; btn.disabled=true;
  postFeedback({run_id:btn.dataset.run,kind:'comment',text:text})
    .then(r=>{ if(!r.ok) throw 0; fb.querySelector('.fb-cmt').innerHTML='<span class="fb-thanks">Thanks for the feedback.</span>'; })
    .catch(()=>{ btn.disabled=false; });
}

// ---- delegated clicks ----
document.addEventListener('click',function(e){
  var t=e.target.closest('.sug,.fb-btn,.fb-cmt-toggle,.fb-cmt-send'); if(!t) return;
  if(t.classList.contains('sug')){ window.CLC_PENDING_TITLE=t.textContent.trim(); }
  else if(t.classList.contains('fb-btn')){ vote(t); }
  else if(t.classList.contains('fb-cmt-toggle')){
    var box=t.closest('.fb').querySelector('.fb-cmt'); box.hidden=!box.hidden;
    if(!box.hidden) box.querySelector('input').focus();
  }
  else if(t.classList.contains('fb-cmt-send')){ sendComment(t); }
});
document.addEventListener('submit',function(e){
  if(e.target.id==='chat-form'){ var i=e.target.querySelector('[name=q]'); if(i) window.CLC_PENDING_TITLE=i.value.trim(); }
});
// htmx lifecycle
document.body.addEventListener('htmx:sseMessage',function(){ renderAll(); scrollDown(); });
document.body.addEventListener('htmx:afterRequest',function(e){
  var cfg=e.detail&&e.detail.requestConfig;
  if(cfg&&e.detail.successful&&(cfg.path||'').indexOf('/send')!==-1){
    var es=document.getElementById('empty-state'); if(es) es.remove();
    var f=document.getElementById('chat-form'); if(f) f.reset();
    upsertThread(window.CLC_PENDING_TITLE);
  }
  scrollDown();
});
function init(){ applySidebar(); renderThreads(); startObserver(); renderAll(); scrollDown(); }
if(document.readyState==='loading') window.addEventListener('DOMContentLoaded',init); else init();
"""

# The session cookie carries the chat thread id; its signing key must never be a
# value an attacker could know. Prefer the configured SESSION_SECRET; if unset,
# fall back to an ephemeral per-process key (cookies simply don't survive a
# restart) rather than a hardcoded constant that would let anyone forge a session
# and read another user's thread. Set SESSION_SECRET in any real deployment.
_SESSION_SECRET = os.getenv("SESSION_SECRET")
if not _SESSION_SECRET:
    _SESSION_SECRET = secrets.token_urlsafe(32)
    print(
        "⚠️  SESSION_SECRET not set — using an ephemeral signing key. "
        "Set SESSION_SECRET in deployment so sessions survive restarts."
    )

app, rt = fast_app(
    hdrs=HEAD,
    htmx=True,
    pico=False,
    surreal=False,
    secret_key=_SESSION_SECRET,
    title="Chat LangChain Lite",
)


# ── Content helpers ───────────────────────────────────────────────────────────
def _avatar(role: str) -> FT:
    return Div("🧑" if role == "user" else "💬", cls="avatar")


def _msg_text(content) -> str:
    """Flatten message content (str or Anthropic-style text blocks) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text") or "" for b in content if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


def user_bubble(text: str) -> FT:
    # `text` is an FT text node → auto-escaped (no XSS).
    return Div(_avatar("user"), Div(Div(text, cls="bubble"), cls="msg-main"), cls="msg user")


def _thumb(icon: str, score: str, run_id: str, title: str, selected: bool) -> FT:
    """One 👍/👎 button. The selected one is highlighted AND disabled (you switch
    by clicking the other), giving a clear, persistent indication of your vote."""
    if selected:
        return Button(
            icon, cls="fb-btn sel", title=title, data_score=score, data_run=run_id, disabled=True
        )
    return Button(icon, cls="fb-btn", title=title, data_score=score, data_run=run_id)


def fb_bar(run_id: str, score: int | None = None) -> FT:
    """Feedback (👍/👎 + optional comment) + trace link for one response.

    Used identically for live and historical messages. `score` (1/0/None) is the
    user's current vote, recovered from LangSmith for history so the selection
    persists across thread switches. Votes POST to the same-origin /feedback
    endpoint, which upserts one record per (run, key).
    """
    return Div(
        _thumb("👍", "1", run_id, "Helpful", score == 1),
        _thumb("👎", "0", run_id, "Not helpful", score == 0),
        Button("💬 Comment", cls="fb-cmt-toggle", title="Add a comment"),
        A("↗ Trace", cls="fb-trace", href=f"/trace/{run_id}", target="_blank", rel="noopener"),
        Div(
            Input(cls="fb-cmt-input", placeholder="Add a comment (optional)…"),
            Button("Send", cls="fb-cmt-send", data_run=run_id),
            cls="fb-cmt",
            hidden=True,
        ),
        cls="fb",
    )


def assistant_live(run_id: str) -> FT:
    """A live assistant message that *joins* an already-created run over SSE.

    The run is created once by POST /send; the SSE endpoint only joins its stream,
    so EventSource reconnects re-attach to the same run (no duplicate turns). The
    feedback/trace bar is swapped into the actions slot when the stream completes.
    """
    body = Div(Span(cls="cursor"), cls="bubble md-live", sse_swap="token", hx_swap="innerHTML")
    actions = Div(sse_swap="actions", hx_swap="innerHTML")
    return Div(
        _avatar("assistant"),
        Div(body, actions, cls="msg-main"),
        cls="msg assistant",
        hx_ext="sse",
        sse_connect=f"/stream?run={run_id}",
        sse_close="done",
    )


def assistant_static(run_id: str | None, text: str, score: int | None = None) -> FT:
    """A rendered (historical) assistant message with the same feedback/trace bar."""
    main = [Div(text, cls="bubble md-live")]
    if run_id:
        main.append(fb_bar(run_id, score))
    return Div(_avatar("assistant"), Div(*main, cls="msg-main"), cls="msg assistant")


def empty_state() -> FT:
    cards = [
        Button(
            text,
            cls="sug",
            hx_post="/send",
            hx_vals=json.dumps({"q": text}),
            hx_target="#messages",
            hx_swap="beforeend",
            hx_disabled_elt="this",
        )
        for text in SUGGESTIONS
    ]
    return Div(
        Div(
            Img(src=LOGO),
            H1("Chat LangChain Lite"),
            P("Ask anything about LangChain, LangGraph, LangSmith, and Deep Agents"),
            cls="empty",
        ),
        Div("Try one of these", cls="sug-label"),
        Div(*cards, cls="sug-grid"),
        id="empty-state",
    )


def sidebar() -> FT:
    return Aside(
        Div(
            Div(Img(src=LOGO), Span("Chat LangChain Lite"), cls="sb-brand"),
            Button("«", cls="icon-btn", title="Collapse sidebar", onclick="toggleSidebar()"),
            cls="sb-head",
        ),
        A("✏️  New chat", href="/?new=1", cls="sb-new"),
        Div("Recent", cls="sb-section"),
        Nav(id="sb-threads"),  # filled from localStorage
        Div("Conversations are stored in your browser.", cls="sb-foot"),
        cls="sidebar",
    )


def topbar() -> FT:
    return Header(
        Button("☰", cls="icon-btn", title="Toggle sidebar", onclick="toggleSidebar()"),
        Div(Img(src=LOGO), Span("Chat LangChain Lite"), cls="brand"),
        cls="topbar",
    )


def composer() -> FT:
    return Div(
        Div(
            Form(
                Input(
                    name="q",
                    placeholder="Message Chat LangChain Lite…",
                    autocomplete="off",
                    required=True,
                    cls="chat-input",
                ),
                Button("➤", cls="send-btn", type="submit", title="Send"),
                hx_post="/send",
                hx_target="#messages",
                hx_swap="beforeend",
                hx_disabled_elt="find .send-btn",
                cls="chat-form",
                id="chat-form",
            ),
            Div("Responses are traced to LangSmith. Rate them with 👍 / 👎.", cls="composer-hint"),
            cls="composer-inner",
        ),
        cls="composer",
    )


def page(body_children: list[FT], thread_id: str) -> FT:
    return Div(
        sidebar(),
        Main(
            topbar(),
            Div(Div(*body_children, id="messages", cls="chat-inner"), cls="chat"),
            composer(),
            cls="main",
        ),
        Script(f"window.CLC_THREAD={json.dumps(thread_id)};"),
        Script(CLIENT_JS),
        cls="app",
        id="app",
    )


# ── Routes ────────────────────────────────────────────────────────────────────
@rt("/")
async def index(session, new: str = "", thread: str = ""):
    if new:
        session["thread"] = str(uuid.uuid4())
    elif thread and _UUID_RE.match(thread):
        session["thread"] = thread
    elif "thread" not in session:
        session["thread"] = str(uuid.uuid4())

    thread_id = session["thread"]
    # Render history for the active session thread on any load except an explicit
    # "new" — so a plain refresh (no ?thread=) shows the ongoing conversation
    # instead of the empty state and silently appending into a hidden thread.
    body: list[FT] = [] if new else await _history_bubbles(thread_id)
    if not body:
        body = [empty_state()]
    return page(body, thread_id)


@rt("/send")
async def send(session, q: str = ""):
    q = (q or "").strip()
    if "thread" not in session:
        session["thread"] = str(uuid.uuid4())
    if not q:
        return ""
    thread_id = session["thread"]
    # Create the run ONCE here. The assistant bubble then joins this run's stream
    # over SSE, so EventSource reconnects re-attach instead of starting new runs.
    try:
        # Name + tag UI traffic consistently with the scripted path (agent._config),
        # so the chat UI's runs aren't named after the bare graph ("chat_langchain_lite").
        # `run_name` is a valid RunnableConfig field the graph honors; the SDK's
        # Config TypedDict just omits it, hence the ignore.
        run = await get_client(url=_api_url()).runs.create(  # ty: ignore[no-matching-overload]
            thread_id,
            ASSISTANT_ID,
            input={"messages": [{"role": "user", "content": q}]},
            stream_mode="messages-tuple",
            stream_resumable=True,
            if_not_exists="create",
            metadata={"demo": "true", "demo_type": APP_SLUG},
            config={
                "run_name": f"{APP_SLUG}-demo",
                "tags": ["engine-demo", CONTEXT_HUB_REPO],
            },
        )
    except Exception:
        return (
            user_bubble(q),
            assistant_static(None, "⚠️ Couldn't reach the agent. Please try again."),
        )
    return (user_bubble(q), assistant_live(run["run_id"]))


@rt("/stream")
async def stream(session, run: str = ""):
    thread_id = session.get("thread") or ""
    run_id = run if _UUID_RE.match(run or "") else ""

    async def gen():
        acc = ""
        if not run_id or not thread_id:
            yield sse_message("⚠️ Invalid run.", event="token")
            yield sse_message(
                "[DONE]", event="done"
            )  # non-empty so the event dispatches (see below)
            return
        try:
            client = get_client(url=_api_url())
            async for part in client.runs.join_stream(
                thread_id, run_id, stream_mode="messages-tuple"
            ):
                if part.event != "messages":
                    continue
                data = part.data
                if not isinstance(data, list) or not data:
                    continue
                msg = data[0]
                if not isinstance(msg, dict) or msg.get("type") != "AIMessageChunk":
                    continue
                acc += _msg_text(msg.get("content"))
                if acc:
                    yield sse_message(acc, event="token")
        except Exception:
            # Each token frame replaces the bubble, so append to acc rather than
            # emitting the warning alone — otherwise a mid-stream failure would
            # wipe the partial answer already shown.
            warning = "\n\n⚠️ The response was interrupted. Please try again."
            yield sse_message((acc + warning) if acc else warning.strip(), event="token")
        # Feedback + trace bar once the response is complete.
        yield sse_message(fb_bar(run_id), event="actions")
        # The payload must be non-empty: an SSE frame with no `data:` line is not
        # dispatched by the browser EventSource, so an empty "done" would never
        # fire — `sse_close="done"` (assistant_live) wouldn't close the stream, and
        # the auto-reconnect would re-swap the actions slot, wiping the user's vote
        # / open comment box. The sentinel is inert (nothing has sse-swap="done").
        yield sse_message("[DONE]", event="done")

    return EventStream(gen())


@rt("/feedback")
async def feedback(run_id: str = "", kind: str = "", value: str = "", text: str = ""):
    """Submit user feedback for a response (same-origin; upserts one record).

    A deterministic feedback_id per (run, key) means repeated/switched votes
    UPDATE the single record rather than creating duplicates.
    """
    if not _UUID_RE.match(run_id or ""):
        return PlainTextResponse("Invalid run id.", status_code=400)
    if kind == "comment" and not (text or "").strip():
        return PlainTextResponse("", status_code=204)
    if kind not in ("score", "comment"):
        return PlainTextResponse("Unknown feedback kind.", status_code=400)

    def _submit() -> None:
        # The sync LangSmith client blocks; run it off the event loop (langgraph
        # dev rejects blocking calls inside async routes).
        if kind == "score":
            _ensure_score_config()
            score = 1 if value == "1" else 0
            _ls().create_feedback(
                run_id, SCORE_KEY, score=score, feedback_id=_feedback_id(run_id, SCORE_KEY)
            )
        else:
            _ls().create_feedback(
                run_id,
                COMMENT_KEY,
                comment=text.strip(),
                feedback_id=_feedback_id(run_id, COMMENT_KEY),
            )

    try:
        await asyncio.to_thread(_submit)
    except Exception:
        return PlainTextResponse("Feedback failed.", status_code=502)
    return PlainTextResponse("", status_code=204)


@rt("/trace/{run_id}")
async def view_trace(run_id: str):
    """Redirect to a response's LangSmith trace (open-redirect guarded)."""
    if not _UUID_RE.match(run_id or ""):
        return PlainTextResponse("Invalid run id.", status_code=400)

    def _read_url() -> str | None:
        try:
            return _ls().read_run(run_id).url
        except Exception:
            return None

    url = None
    for attempt in range(4):  # the trace may still be flushing for a few seconds
        url = await asyncio.to_thread(_read_url)
        if url:
            break
        if attempt < 3:
            await asyncio.sleep(0.8)
    host = urlparse(url or "").hostname or ""
    if not url or not any(host == h or host.endswith("." + h) for h in _TRACE_HOST_SUFFIX):
        return PlainTextResponse("Trace not available yet — try again shortly.", status_code=404)
    return RedirectResponse(url)


# ── Feedback / history helpers ────────────────────────────────────────────────
def _feedback_id(run_id: str, key: str) -> str:
    """Deterministic feedback id per (run, key) so submissions upsert one record."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"chat-lc-lite:{run_id}:{key}"))


_SCORE_CONFIG_SET = False
_SCORE_CONFIG: FeedbackConfig = {
    "type": "categorical",
    "categories": [{"value": 1, "label": "👍 Helpful"}, {"value": 0, "label": "👎 Not helpful"}],
}


def _ensure_score_config() -> None:
    """Register user_score as a categorical (👍/👎) feedback config, once."""
    global _SCORE_CONFIG_SET
    if _SCORE_CONFIG_SET:
        return
    _SCORE_CONFIG_SET = True
    try:
        _ls().update_feedback_config(SCORE_KEY, feedback_config=_SCORE_CONFIG)
    except Exception:
        with contextlib.suppress(Exception):  # best-effort; scoring works without it
            _ls().create_feedback_config(SCORE_KEY, feedback_config=_SCORE_CONFIG)


async def _thread_run_ids(client, thread_id: str) -> list[str]:
    """Successful run ids for a thread, oldest-first (one per assistant turn)."""
    try:
        runs = await client.runs.list(thread_id, limit=100, status="success")
    except Exception:
        return []
    runs = [r for r in runs if isinstance(r, dict) and r.get("run_id")]
    runs.sort(key=lambda r: r.get("created_at") or "")
    return [r["run_id"] for r in runs]


def _thread_votes(run_ids: list[str]) -> dict[str, int]:
    """Current 👍/👎 score per run, recovered from LangSmith (for history)."""
    if not run_ids:
        return {}
    try:
        fbs = _ls().list_feedback(run_ids=run_ids, feedback_key=[SCORE_KEY])
        return {str(f.run_id): int(f.score) for f in fbs if f.score is not None}
    except Exception:
        return {}


def _count_assistant_turns(messages: list) -> int:
    """Number of assistant bubbles a message list collapses into.

    Mirrors _history_bubbles' grouping: consecutive AI-text messages form one
    bubble, broken by a human-text message. Kept in sync so the count can be
    compared against the run-id list before binding them positionally.
    """
    turns = 0
    pending = False
    for m in messages:
        if not isinstance(m, dict):
            continue
        text = _msg_text(m.get("content"))
        if m.get("type") == "human" and text:
            if pending:
                turns += 1
                pending = False
        elif m.get("type") == "ai" and text:
            pending = True
    return turns + 1 if pending else turns


async def _history_bubbles(thread_id: str) -> list[FT]:
    """Render a thread's prior messages from the graph's own state.

    Each assistant turn is keyed to its run (oldest-first) so it gets the same
    feedback/trace bar as a live message, with the stored vote pre-selected.
    """
    if not _UUID_RE.match(thread_id or ""):
        return []
    client = get_client(url=_api_url())
    try:
        state = await client.threads.get_state(thread_id)
    except Exception:
        return []
    values = state.get("values") if isinstance(state, dict) else None
    messages = (values or {}).get("messages") if isinstance(values, dict) else None
    if not isinstance(messages, list):
        return []

    run_ids = await _thread_run_ids(client, thread_id)

    # We map assistant bubbles to run ids positionally (oldest-first). That only
    # holds if there's exactly one success run per assistant turn; an errored /
    # interrupted run (excluded from run_ids) would shift every later mapping and
    # attach votes/trace links to the WRONG response. If the counts disagree,
    # drop the run binding entirely — better to show history without feedback
    # bars than to mis-attribute them.
    assistant_turns = _count_assistant_turns(messages)
    if len(run_ids) != assistant_turns:
        run_ids = []

    # _thread_votes uses the *sync* LangSmith client; offload it so we don't make
    # a blocking call inside this async route (langgraph dev rejects those).
    votes = await asyncio.to_thread(_thread_votes, run_ids)

    bubbles: list[FT] = []
    pending_ai: list[str] = []
    turn = 0

    def _flush_ai() -> None:
        nonlocal turn
        if pending_ai:
            rid = run_ids[turn] if turn < len(run_ids) else None
            score = votes.get(rid) if rid else None
            bubbles.append(assistant_static(rid, "\n\n".join(pending_ai), score))
            pending_ai.clear()
            turn += 1

    for m in messages:
        if not isinstance(m, dict):
            continue
        mtype = m.get("type")
        text = _msg_text(m.get("content"))
        if mtype == "human" and text:
            _flush_ai()
            bubbles.append(user_bubble(text))
        elif mtype == "ai" and text:
            # Collapse consecutive assistant messages (preamble + final answer)
            # into one bubble, matching the live single-bubble rendering.
            pending_ai.append(text)
    _flush_ai()
    return bubbles
