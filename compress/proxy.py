"""Compressing OpenAI-compatible proxy for llama-server.

  .venv\\Scripts\\python.exe proxy.py [--port 8081] [--upstream http://127.0.0.1:8080] [--rate 0.33] [--min-words 400]

Long message text is compressed before it reaches the model, so more source fits in the context window and
prefill is shorter:
  * prose  -> LLMLingua-2 token classifier (CPU; keeps the most informative ~rate of tokens, in order)
  * code   -> lossless-ish strip: comments, blank lines, trailing spaces (LLMLingua mangles code)
  * the last paragraph of the final user message (usually the question) is never compressed
Results are cached by content hash, so the same history compresses to the same bytes every turn and
llama-server's prompt cache keeps working.
Per request: add "compress": false, or "compress_rate": 0.5, to the JSON body.
"""
import argparse, hashlib, json, os, re, subprocess, sys, threading, time, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

A = argparse.ArgumentParser()
A.add_argument("--port", type=int, default=8081)
A.add_argument("--upstream", default="http://127.0.0.1:8080")
A.add_argument("--rate", type=float, default=0.33)
A.add_argument("--min-words", type=int, default=400)
A.add_argument("--model", default="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank")
A.add_argument("--offload-min-tokens", type=int, default=0,
               help="prefill on Modal when a request brings at least this many new (uncached) tokens; 0 = off")
A.add_argument("--offload-ctx", type=int, default=131072, help="must equal the local server's -c")
A.add_argument("--offload-kv", default="q4_0", help="must equal the local server's -ctk/-ctv")
A.add_argument("--no-compress", action="store_true", help="only offload, never compress")
ARGS = A.parse_args()

_lock = threading.Lock()
_comp = None
CACHE = {}
STATS = {"in_words": 0, "out_words": 0, "secs": 0.0}


def compressor():
    global _comp
    with _lock:
        if _comp is None:
            from llmlingua import PromptCompressor
            _comp = PromptCompressor(model_name=ARGS.model, use_llmlingua2=True, device_map="cpu")
    return _comp


FENCE = re.compile(r"(```[^\n]*\n.*?```)", re.S)


def strip_code(block):
    head, _, body = block.partition("\n")
    out = []
    for line in body.splitlines():
        s = line.rstrip()
        t = s.lstrip()
        if not t or t.startswith(("#", "//")) and not t.startswith(("#!", "#include", "#define", "#[")):
            continue
        out.append(s)
    return head + "\n" + "\n".join(out)


def _compress_chunk(text, rate):
    key = hashlib.sha256(f"{rate}|{text}".encode()).hexdigest()
    if key not in CACHE:
        r = compressor().compress_prompt(text, rate=rate, force_tokens=["\n", ".", "?", "!", ":", "=", "(", ")"],
                                         force_reserve_digit=True, drop_consecutive=True)
        CACHE[key] = r["compressed_prompt"]
    return CACHE[key]


def compress_prose(text, rate, chunk_words=300):
    """Compress paragraph groups of ~chunk_words separately: a shared document prefix gives identical chunks
    (cache hits, and identical bytes so llama-server's prompt cache is reused)."""
    if len(text.split()) < 40:
        return text
    out, buf, n = [], [], 0
    for p in text.split("\n\n"):
        buf.append(p); n += len(p.split())
        if n >= chunk_words:
            out.append(_compress_chunk("\n\n".join(buf), rate)); buf, n = [], 0
    if buf:
        tail = "\n\n".join(buf)
        out.append(_compress_chunk(tail, rate) if n >= 40 else tail)
    return "\n\n".join(out)


def compress_text(text, rate, keep_tail):
    t0 = time.time()
    tail = ""
    if keep_tail:
        body, sep, tail = text.rstrip().rpartition("\n\n")
        if not sep or len(tail.split()) > 300:
            body, tail = text, ""
    else:
        body = text
    parts = FENCE.split(body)
    out = [strip_code(p) if p.startswith("```") else compress_prose(p, rate) for p in parts]
    res = "".join(out) + ("\n\n" + tail if tail else "")
    STATS["in_words"] += len(text.split()); STATS["out_words"] += len(res.split()); STATS["secs"] += time.time() - t0
    return res


HERE = os.path.dirname(os.path.abspath(__file__))
_off_lock = threading.Lock()
LAST_PROMPT = [""]


def maybe_offload(body):
    """Prefill on Modal when the templated prompt has many tokens the local slot has not seen yet.
    The local slot holds LAST_PROMPT; only the part after the shared prefix would need a local prefill."""
    keys = ("messages", "tools", "tool_choice", "chat_template_kwargs", "reasoning_format")
    req = urllib.request.Request(ARGS.upstream + "/apply-template", json.dumps({k: body[k] for k in keys if k in body}).encode(),
                                 {"Content-Type": "application/json"})
    prompt = json.load(urllib.request.urlopen(req, timeout=120))["prompt"]
    new_chars = len(prompt) - len(os.path.commonprefix([LAST_PROMPT[0], prompt]))
    LAST_PROMPT[0] = prompt
    if new_chars / 3.5 < ARGS.offload_min_tokens:
        return
    pf = os.path.join(HERE, "_offload_prompt.txt")
    open(pf, "w", encoding="utf-8").write(prompt)
    py = os.path.join(HERE, "..", "cloud", ".venv", "Scripts", "python.exe")
    r = subprocess.run([py, os.path.join(HERE, "..", "cloud", "offload.py"), pf, "--n-ctx", str(ARGS.offload_ctx),
                        "--kv", ARGS.offload_kv, "--url", ARGS.upstream], capture_output=True, text=True, timeout=3600)
    os.remove(pf)
    print("offload:", r.stdout.strip().splitlines()[-1] if r.returncode == 0 else "FAILED " + r.stderr[-400:], file=sys.stderr, flush=True)
    body["id_slot"] = 0
    body["cache_prompt"] = True


def rewrite(body):
    if ARGS.no_compress:
        body.pop("compress", None); body.pop("compress_rate", None)
        return body
    if body.pop("compress", True) is False:
        body.pop("compress_rate", None)
        return body
    rate = float(body.pop("compress_rate", ARGS.rate))
    msgs = body.get("messages") or []
    last_user = max((i for i, m in enumerate(msgs) if m.get("role") == "user"), default=-1)
    for i, m in enumerate(msgs):
        c = m.get("content")
        if m.get("role") not in ("user", "system", "tool") or not isinstance(c, str):
            continue
        if len(c.split()) >= ARGS.min_words:
            m["content"] = compress_text(c, rate, keep_tail=(i == last_user))
    return body


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _forward(self, method, data=None):
        req = urllib.request.Request(ARGS.upstream + self.path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            r = urllib.request.urlopen(req, timeout=7200)
        except urllib.error.HTTPError as e:
            r = e
        self.send_response(r.status)
        for k, v in r.headers.items():
            if k.lower() not in ("transfer-encoding", "connection", "content-length"):
                self.send_header(k, v)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        while True:
            chunk = r.read1(65536) if hasattr(r, "read1") else r.read(65536)
            if not chunk:
                break
            self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk)); self.wfile.flush()
        self.wfile.write(b"0\r\n\r\n")

    def do_GET(self):
        if self.path == "/compress/stats":
            b = json.dumps({**STATS, "ratio": STATS["out_words"] / max(1, STATS["in_words"]), "cached": len(CACHE)}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
            return
        self._forward("GET")

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.path.rstrip("/").endswith("/chat/completions"):
            try:
                body = rewrite(json.loads(raw))
                if ARGS.offload_min_tokens:
                    with _off_lock:
                        try:
                            maybe_offload(body)
                        except Exception as e:
                            print("offload failed, prefilling locally:", repr(e), file=sys.stderr)
                raw = json.dumps(body).encode()
            except Exception as e:
                print("compress failed, passing through:", repr(e), file=sys.stderr)
        self._forward("POST", raw)


if __name__ == "__main__":
    if not ARGS.no_compress:
        print(f"loading {ARGS.model} ...", flush=True)
        compressor()
    print(f"compressing proxy on http://127.0.0.1:{ARGS.port} -> {ARGS.upstream} (rate {ARGS.rate})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", ARGS.port), H).serve_forever()
