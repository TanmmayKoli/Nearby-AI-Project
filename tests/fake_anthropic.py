"""Minimal local fake of the Anthropic Messages API, for offline tests.

Returns JSON for normal calls and server-sent events for stream=true, in the
same event format the real API uses, so the real ChatAnthropic client (and
LangChain's chunk merging) runs unmodified. Set WITH_THINKING to put a
thinking block before the text, as claude-sonnet-5-5 does.
"""
import json, threading
from http.server import BaseHTTPRequestHandler, HTTPServer

EXTRACTION = {"category": "plumbing", "category_confidence": 0.9, "new_facts": ["Water heater dripping"],
              "retracted_facts": [], "out_of_scope_reason": None, "cause_unknown": False, "declined_contact": [],
              "remove_fields": [], "urgency": None, "zip": None, "name": None, "contact_phone": None,
              "contact_email": None, "address": None, "property_type": None, "owner_or_renter": None,
              "availability": None, "category_details": [], "safety_flags": []}
QUESTION = "How soon do you need someone to take a look?"
WITH_THINKING = False

def message(text):
    content = ([{"type": "thinking", "thinking": "hmm", "signature": "x"}] if WITH_THINKING else []) + [{"type": "text", "text": text}]
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-sonnet-5-5", "content": content,
            "stop_reason": "end_turn", "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 10}}

def sse(text):
    ev = []
    ev.append(("message_start", {"type": "message_start", "message": {**message(""), "content": [], "stop_reason": None}}))
    i = 0
    if WITH_THINKING:
        ev.append(("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}}))
        ev.append(("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "hmm"}}))
        ev.append(("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "x"}}))
        ev.append(("content_block_stop", {"type": "content_block_stop", "index": 0}))
        i = 1
    ev.append(("content_block_start", {"type": "content_block_start", "index": i, "content_block": {"type": "text", "text": ""}}))
    for word in text.split(" "):
        ev.append(("content_block_delta", {"type": "content_block_delta", "index": i, "delta": {"type": "text_delta", "text": word + " "}}))
    ev.append(("content_block_stop", {"type": "content_block_stop", "index": i}))
    ev.append(("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": 10}}))
    ev.append(("message_stop", {"type": "message_stop"}))
    return "".join(f"event: {e}\ndata: {json.dumps(d)}\n\n" for e, d in ev).encode()

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        text = json.dumps(EXTRACTION) if "output_config" in body else QUESTION
        if body.get("stream"):
            self.send_response(200); self.send_header("content-type", "text/event-stream"); self.end_headers()
            self.wfile.write(sse(text))
        else:
            payload = json.dumps(message(text)).encode()
            self.send_response(200); self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload))); self.end_headers(); self.wfile.write(payload)

def start(port: int = 0) -> HTTPServer:
    """Start on a free port (port=0); the URL is f"http://127.0.0.1:{srv.server_port}"."""
    srv = HTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv
