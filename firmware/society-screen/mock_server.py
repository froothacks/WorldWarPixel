#!/usr/bin/env python3
"""Mock WorldWarPixel server for testing the society-screen firmware.

Implements the device protocol from the handover:
  GET  /devices/me            -> device/society/run JSON
  GET  /devices/canvas?w&h    -> w*h palette indices (octet-stream)
  POST /devices/voice         -> {"transcript","murmur"}
  WS   /devices/ws            -> hello, state (~2s), migration flashes,
                                 voice.start/binary/voice.end streaming

Stdlib only: no pip installs needed.

Usage:
  DEVICE_TOKENS="alpha:0,bravo:1,charlie:2" python3 mock_server.py [port]
Default tokens: alpha=0, bravo=1, charlie=2
"""

import base64
import hashlib
import json
import os
import random
import socket
import struct
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

TOKENS = {}
for part in os.environ.get("DEVICE_TOKENS", "alpha:0,bravo:1,charlie:2").split(","):
    if ":" in part:
        tok, soc = part.split(":", 1)
        TOKENS[tok.strip()] = int(soc)

PALETTE = [
    "#ffffff", "#e4e4e4", "#888888", "#222222",
    "#ff4500", "#ffa800", "#ffd635", "#00a368",
    "#7eed56", "#2450a4", "#3690ea", "#51e9f4",
    "#811e9f", "#b44ac0", "#ff99aa", "#9c6926",
]

SOCIETIES = [
    {"id": 0, "name": "Ember", "color": "#ff4500", "members": 7},
    {"id": 1, "name": "Tide",  "color": "#3690ea", "members": 5},
    {"id": 2, "name": "Moss",  "color": "#00a368", "members": 6},
]

THREADS = [
    [{"title": "Push the eastern border at dawn", "author": "Ana Souza", "replies": 4},
     {"title": "Who took my pixel at (12,40)?",   "author": "Ben Okafor", "replies": 2},
     {"title": "Defence grid proposal v2",        "author": "C. Lindqvist", "replies": 7}],
    [{"title": "Tide rising: claim the coast",    "author": "D. Marchetti", "replies": 5},
     {"title": "Trade pixels for protection?",    "author": "Ela Nowak", "replies": 1},
     {"title": "Water theme for our quadrant",    "author": "F. Haddad", "replies": 3}],
    [{"title": "Grow moss over the grey zone",    "author": "G. Tanaka", "replies": 6},
     {"title": "Slow and steady placement",       "author": "H. Virtanen", "replies": 2},
     {"title": "Forest mural progress pics",      "author": "I. Petrova", "replies": 9}],
]

MURMURS = [
    None,
    "people here have been saying the east border needs defenders",
    "people here have been saying someone should paint a flag",
    "people here have been saying they miss the old leader",
]

TRANSCRIPTS = [
    "we should all paint the same symbol in the corner",
    "someone is erasing our border, watch the north edge",
    "let us make the flag bigger before nightfall",
]

AGENTS = ["Ana Souza", "Ben Okafor", "C. Lindqvist", "D. Marchetti",
          "Ela Nowak", "F. Haddad", "G. Tanaka", "H. Virtanen"]

run_id = "effa98e9"


def society_state(soc_id):
    soc = dict(SOCIETIES[soc_id])
    soc["joined"] = random.randint(0, 2)
    soc["left"] = random.randint(0, 1)
    return {
        "type": "state",
        "run": run_id,
        "society": soc,
        "societies": [{"id": s["id"], "name": s["name"], "members": s["members"]}
                      for s in SOCIETIES],
        "threads": THREADS[soc_id],
        "murmur": murmur_for(soc_id),
        "voice": voice_state.get(soc_id, {"status": "idle", "transcript": "", "error": None}),
    }


murmurs = {}
voice_state = {}
voice_audio = {}


def murmur_for(soc_id):
    return murmurs.get(soc_id)


# ---------- minimal websocket framing ----------

def ws_accept(key):
    return base64.b64encode(
        hashlib.sha1((key + WS_GUID).encode()).digest()).decode()


def ws_send_frame(conn, opcode, payload):
    header = bytearray([0x80 | opcode])
    n = len(payload)
    if n < 126:
        header.append(n)
    elif n < 65536:
        header.append(126)
        header += struct.pack(">H", n)
    else:
        header.append(127)
        header += struct.pack(">Q", n)
    conn.sendall(bytes(header) + payload)


def ws_send_text(conn, obj):
    ws_send_frame(conn, 0x1, json.dumps(obj).encode())


def ws_recv_frame(conn):
    """Returns (opcode, payload) or None on close/error."""
    try:
        hdr = conn.recv(2)
        if len(hdr) < 2:
            return None
        opcode = hdr[0] & 0x0F
        masked = hdr[1] & 0x80
        n = hdr[1] & 0x7F
        if n == 126:
            n = struct.unpack(">H", conn.recv(2))[0]
        elif n == 127:
            n = struct.unpack(">Q", conn.recv(8))[0]
        mask = conn.recv(4) if masked else b""
        payload = b""
        while len(payload) < n:
            chunk = conn.recv(n - len(payload))
            if not chunk:
                return None
            payload += chunk
        if masked:
            payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        return opcode, payload
    except OSError:
        return None


# ---------- http handler ----------

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write("  %s\n" % (fmt % args))

    def _token(self):
        auth = self.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            return auth[7:]
        qs = parse_qs(urlparse(self.path).query)
        return qs.get("token", [None])[0]

    def _society(self):
        tok = self._token()
        return TOKENS.get(tok)

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        soc = self._society()
        if soc is None:
            return self._json(401, {"error": "bad token"})
        if path == "/devices/me":
            return self._json(200, {
                "device": "screen-%d" % soc,
                "society": soc,
                "run": run_id,
            })
        if path == "/devices/canvas":
            qs = parse_qs(urlparse(self.path).query)
            w = min(int(qs.get("w", [160])[0]), 320)
            h = min(int(qs.get("h", [100])[0]), 240)
            data = bytes(random.choice([0, 0, 0, 1, 2, 3,
                                        4 + soc * 5]) for _ in range(w * h))
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("X-Width", str(w))
            self.send_header("X-Height", str(h))
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if path == "/devices/ws":
            return self._ws(soc)
        self._json(404, {"error": "not found"})

    def do_POST(self):
        path = urlparse(self.path).path
        soc = self._society()
        if soc is None:
            return self._json(401, {"error": "bad token"})
        if path == "/devices/voice":
            n = int(self.headers.get("Content-Length", 0))
            self.rfile.read(n)  # discard PCM
            transcript = random.choice(TRANSCRIPTS)
            murmurs[soc] = "people here have been saying " + transcript
            return self._json(200, {"transcript": transcript,
                                    "murmur": murmurs[soc]})
        self._json(404, {"error": "not found"})

    # -- websocket --
    def _ws(self, soc):
        key = self.headers.get("Sec-WebSocket-Key")
        if not key:
            return self._json(400, {"error": "not a websocket request"})
        self.send_response(101, "Switching Protocols")
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", ws_accept(key))
        self.end_headers()

        conn = self.request
        conn.settimeout(1.0)
        write_lock = threading.Lock()
        alive = [True]

        def send(obj):
            if not alive[0]:
                return
            try:
                with write_lock:
                    ws_send_text(conn, obj)
            except OSError:
                alive[0] = False

        def pusher():
            n = 0
            while alive[0]:
                time.sleep(2.0)
                send(society_state(soc))
                n += 1
                if n % 6 == 0 and alive[0]:   # migration flash ~every 12s
                    frm, to = random.sample(range(3), 2)
                    send({"type": "migration", "agent": random.choice(AGENTS),
                          "from": frm, "to": to})

        threading.Thread(target=pusher, daemon=True).start()
        send({"type": "hello", "device": "screen-%d" % soc,
              "society": soc, "palette": PALETTE})

        voice_state[soc] = {"status": "idle", "transcript": "", "error": None}
        voice_audio[soc] = 0

        while alive[0]:
            frame = ws_recv_frame(conn)
            if frame is None:
                break
            opcode, payload = frame
            if opcode == 0x8:            # close
                break
            if opcode == 0x2:            # binary: PCM audio
                voice_audio[soc] = voice_audio.get(soc, 0) + len(payload)
                continue
            if opcode != 0x1:
                continue
            try:
                msg = json.loads(payload)
            except ValueError:
                continue
            t = msg.get("type")
            if t == "ping":
                send({"type": "pong"})
            elif t == "voice.start":
                voice_audio[soc] = 0
                voice_state[soc] = {"status": "listening",
                                    "transcript": "", "error": None}
                send(society_state(soc))
            elif t in ("voice.end", "voice.cancel"):
                if t == "voice.cancel":
                    voice_state[soc] = {"status": "idle",
                                        "transcript": "", "error": None}
                    send(society_state(soc))
                    continue
                secs = voice_audio[soc] / 32000.0
                voice_state[soc] = {"status": "transcribing",
                                    "transcript": "", "error": None}
                send(society_state(soc))

                def finish(secs=secs):
                    time.sleep(1.5)
                    transcript = random.choice(TRANSCRIPTS)
                    if secs < 0.3:
                        transcript = "(too short - server heard %.1fs)" % secs
                    murmurs[soc] = "people here have been saying " + transcript
                    voice_state[soc] = {"status": "done",
                                        "transcript": transcript, "error": None}
                    send(society_state(soc))

                threading.Thread(target=finish, daemon=True).start()

        alive[0] = False
        voice_state.pop(soc, None)
        print("ws closed (society %d)" % soc)


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print("mock WorldWarPixel server on 0.0.0.0:%d" % port)
    print("tokens:", ", ".join("%s->society %d" % (t, s) for t, s in TOKENS.items()))
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
