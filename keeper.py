"""Thinking keeper: a front door for llama-server that puts Inkling's thinking back into the chat history.

  python keeper.py [--port 8080] [--upstream http://127.0.0.1:8079] [--store keeper-cache.jsonl]

Inkling is trained to see its own earlier thinking in the history. llama-server returns the thinking as
`reasoning_content` and renders it back into the prompt only when the client sends that field again. Most chat
apps and agent harnesses drop it, and the model then works from a history it was never trained on.

The keeper remembers the thinking of every reply and adds it back to assistant messages that arrive without it.
Everything else (web UI, other endpoints, streaming) passes through untouched. Standard library only.
  GET /keeper/stats  -> counters
"""
import argparse, hashlib, http.client, json, os, re, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

A = argparse.ArgumentParser()
A.add_argument("--port", type=int, default=8080)
A.add_argument("--upstream", default="http://127.0.0.1:8079")
A.add_argument("--store", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "keeper-cache.jsonl"))
A.add_argument("--max-entries", type=int, default=4000)
ARGS = A.parse_args()
UP = urlsplit(ARGS.upstream)

LOCK = threading.Lock()
STORE = {}      # key -> thinking text
STATS = {"chat_requests": 0, "replies_stored": 0, "messages_restored": 0, "messages_already_had_thinking": 0,
         "messages_had_no_thinking": 0, "messages_not_found": 0}
THINK = re.compile(r"^\s*<think(?:ing)?>(.*?)</think(?:ing)?>\s*", re.S)
ALT_KEYS = ("reasoning", "reasoning_text", "thinking")
HOP = {"connection", "keep-alive", "transfer-encoding", "content-length", "te", "upgrade", "proxy-connection", "host"}


def h(s):
    return hashlib.sha256(s.encode("utf-8", "replace")).hexdigest()[:32]


def text_of(c):
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(p if isinstance(p, str) else (p.get("text") or "") if isinstance(p, dict) else "" for p in c)
    return ""


def calls_of(m):
    out = []
    for tc in m.get("tool_calls") or []:
        fn = tc.get("function") or {}
        a = fn.get("arguments")
        if isinstance(a, str):
            try:
                a = json.loads(a)
            except ValueError:
                pass
        out.append([fn.get("name"), a])
    return out


def sig(m):
    """What a message says, without its thinking. Whitespace at the ends is ignored: clients often trim it."""
    return json.dumps([m.get("role"), text_of(m.get("content")).strip(), calls_of(m)], sort_keys=True, ensure_ascii=False)


def keys_for(prefix, m):
    """Most specific first: same reply after the same history, then the tool call ids, then the reply text alone."""
    s = sig(m)
    ks = ["p:" + h(prefix + s)]
    ks += ["t:" + tc["id"] for tc in m.get("tool_calls") or [] if isinstance(tc.get("id"), str) and tc["id"]]
    if len(text_of(m.get("content")).strip()) >= 24 or m.get("tool_calls"):
        ks.append("c:" + h(s))
    return ks


def remember(keys, thinking, persist=True):
    with LOCK:
        for k in keys:
            STORE[k] = thinking
        if persist:
            STATS["replies_stored"] += 1
            try:
                with open(ARGS.store, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"k": keys, "r": thinking}, ensure_ascii=False) + "\n")
            except OSError as e:
                print("keeper: cannot write store:", e, file=sys.stderr)


def load_store():
    if not os.path.exists(ARGS.store):
        return
    rows = []
    with open(ARGS.store, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    if len(rows) > ARGS.max_entries:
        rows = rows[-ARGS.max_entries:]
        with open(ARGS.store, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    for r in rows:
        remember(r["k"], r["r"], persist=False)


def restore(body):
    """Add thinking to assistant messages that lack it. Returns the running history digest for remember()."""
    prefix = ""
    for m in body.get("messages") or []:
        if not isinstance(m, dict):
            continue
        if m.get("role") == "assistant":
            if m.get("reasoning_content"):
                STATS["messages_already_had_thinking"] += 1
            else:
                found = None
                for k in ALT_KEYS:                       # other names clients use for the same field
                    if isinstance(m.get(k), str) and m[k]:
                        found = m.pop(k)
                        break
                if not found and isinstance(m.get("reasoning_details"), list):
                    found = "".join(d.get("text") or "" for d in m["reasoning_details"] if isinstance(d, dict)) or None
                if not found and isinstance(m.get("content"), str):
                    t = THINK.match(m["content"])        # thinking left inline in the text
                    if t:
                        found, m["content"] = t.group(1).strip(), m["content"][t.end():]
                if not found:
                    with LOCK:
                        found = next((STORE[k] for k in keys_for(prefix, m) if k in STORE), None)
                if found:
                    m["reasoning_content"] = found
                    STATS["messages_restored"] += 1
                elif found == "":
                    STATS["messages_had_no_thinking"] += 1
                else:
                    STATS["messages_not_found"] += 1
        prefix = h(prefix + sig(m))
    return prefix


class Reply:
    """Collects one assistant reply from a streamed (SSE) or plain JSON response."""

    def __init__(self):
        self.buf = b""
        self.msg = {"role": "assistant", "content": "", "reasoning_content": "", "tool_calls": []}

    def _delta(self, d):
        for k in ("content", "reasoning_content"):
            if isinstance(d.get(k), str):
                self.msg[k] += d[k]
        for tc in d.get("tool_calls") or []:
            i = tc.get("index", 0)
            while len(self.msg["tool_calls"]) <= i:
                self.msg["tool_calls"].append({"id": "", "function": {"name": "", "arguments": ""}})
            cur = self.msg["tool_calls"][i]
            cur["id"] = tc.get("id") or cur["id"]
            fn = tc.get("function") or {}
            cur["function"]["name"] += fn.get("name") or ""
            cur["function"]["arguments"] += fn.get("arguments") or ""

    def feed_sse(self, chunk):
        self.buf += chunk
        while b"\n" in self.buf:
            line, self.buf = self.buf.split(b"\n", 1)
            line = line.strip()
            if not line.startswith(b"data:") or line[5:].strip() == b"[DONE]":
                continue
            try:
                ch = (json.loads(line[5:]).get("choices") or [{}])[0]
            except ValueError:
                continue
            self._delta(ch.get("delta") or {})

    def feed_json(self, raw):
        try:
            m = (json.loads(raw).get("choices") or [{}])[0].get("message") or {}
        except ValueError:
            return
        self.msg.update({"content": m.get("content") or "", "reasoning_content": m.get("reasoning_content") or "",
                         "tool_calls": m.get("tool_calls") or []})

    def store(self, prefix):
        # replies without thinking are remembered too (as ""), so the stats can tell them from real misses
        if self.msg["reasoning_content"] or self.msg["content"] or self.msg["tool_calls"]:
            remember(keys_for(prefix, self.msg), self.msg["reasoning_content"])


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send_json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _proxy(self):
        if self.command == "GET" and self.path == "/keeper/stats":
            with LOCK:
                return self._send_json(200, {**STATS, "store_keys": len(STORE)})
        n = int(self.headers.get("Content-Length") or 0)
        data = self.rfile.read(n) if n else None
        reply = prefix = None
        if self.command == "POST" and data and self.path.split("?")[0].rstrip("/").endswith("/chat/completions"):
            try:
                body = json.loads(data)
                prefix = restore(body)
                data = json.dumps(body, ensure_ascii=False).encode()
                reply = Reply()
                STATS["chat_requests"] += 1
            except Exception as e:                       # never block a request because of the keeper
                print("keeper: passing through unchanged:", repr(e), file=sys.stderr)
                reply = None
        hdrs = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
        if reply:
            hdrs["Accept-Encoding"] = "identity"
        if data is not None:
            hdrs["Content-Length"] = str(len(data))
        conn = http.client.HTTPConnection(UP.hostname, UP.port or 80, timeout=None)
        try:
            try:
                conn.request(self.command, self.path, body=data, headers=hdrs)
                r = conn.getresponse()
            except OSError as e:
                return self._send_json(502, {"error": {"message": f"keeper: llama-server is not reachable at {ARGS.upstream} ({e})"}})
            sse = "text/event-stream" in (r.getheader("Content-Type") or "")
            self.send_response(r.status)
            for k, v in r.getheaders():
                if k.lower() not in HOP:
                    self.send_header(k, v)
            if sse or r.getheader("Content-Length") is None:
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                while True:
                    chunk = r.read1(65536)
                    if not chunk:
                        break
                    if reply and sse:
                        reply.feed_sse(chunk)
                    elif reply:
                        reply.buf += chunk
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
                if reply and not sse:
                    reply.feed_json(reply.buf)
            else:
                raw = r.read()
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                if reply:
                    reply.feed_json(raw)
            if reply and r.status == 200:
                reply.store(prefix)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            self.close_connection = True                 # client went away: closing upstream stops the generation
        finally:
            conn.close()

    do_GET = do_POST = do_PUT = do_DELETE = do_OPTIONS = do_HEAD = do_PATCH = _proxy


if __name__ == "__main__":
    load_store()
    print(f"thinking keeper on http://127.0.0.1:{ARGS.port} -> {ARGS.upstream} ({len(STORE)} keys loaded)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", ARGS.port), H).serve_forever()
