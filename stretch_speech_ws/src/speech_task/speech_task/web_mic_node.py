"""Phone/browser front end for the speech pipeline: hold-to-talk with the phone's mic, or type.

  ros2 run speech_task web_mic                       # https://<this-pc>:8443  (started by speech_pipeline.launch.py)
  ros2 run speech_task web_mic --ros-args -p port:=8443 -p https:=true

Voice: the page records while the button is held and POSTs 16 kHz int16 mono audio, which is published as an
AudioUtterance on /speech/utterance, so it goes through the same Whisper transcriber and interpreter as the robot mic.
Text: published as a Transcript (confidence 1.0) on /speech/transcript, straight to the interpreter.
The page shows what the transcriber heard (/speech/transcript) and the robot's reply (/speech/command).

Phones only allow the microphone on https pages, so the node serves https with a self-signed certificate it
creates on first start (~/.config/speech_web/). The phone shows a certificate warning once: accept it.
Anyone on the same network who opens the page can command the robot.
"""
import array
import json
import os
import socket
import ssl
import subprocess
import threading
import time
import uuid
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import rclpy
from rclpy.node import Node

from speech_task_interfaces.msg import AudioUtterance, TaskCommand, Transcript

STATUS = {TaskCommand.STATUS_EXECUTE: 'execute', TaskCommand.STATUS_CLARIFY: 'clarify',
          TaskCommand.STATUS_INFEASIBLE: 'infeasible', TaskCommand.STATUS_IGNORE: 'ignore'}


class Hub:
    """Recent transcripts and commands, for the page to poll."""

    def __init__(self, keep=200):
        self.items, self.seq, self.cond = deque(maxlen=keep), 0, threading.Condition()

    def add(self, item):
        with self.cond:
            self.seq += 1
            self.items.append({**item, 'seq': self.seq, 't': time.time()})
            self.cond.notify_all()

    def since(self, seq, wait=20.0):
        with self.cond:
            if self.seq <= seq:
                self.cond.wait(timeout=wait)
            return self.seq, [i for i in self.items if i['seq'] > seq]


def local_ips():
    ips = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('10.255.255.255', 1))
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        out = subprocess.run(['hostname', '-I'], capture_output=True, text=True, timeout=2).stdout
        ips.update(ip for ip in out.split() if '.' in ip)
    except (OSError, subprocess.SubprocessError):
        pass
    ips.discard('127.0.0.1')
    return sorted(ips)


def ensure_cert(directory):
    """Self-signed certificate for this machine (created once)."""
    directory = os.path.expanduser(directory)
    cert, key = os.path.join(directory, 'cert.pem'), os.path.join(directory, 'key.pem')
    if os.path.exists(cert) and os.path.exists(key):
        return cert, key
    os.makedirs(directory, exist_ok=True)
    san = ','.join([f'IP:{ip}' for ip in ['127.0.0.1', *local_ips()]] + [f'DNS:{socket.gethostname()}', 'DNS:localhost'])
    subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', '825',
                    '-keyout', key, '-out', cert, '-subj', '/CN=stretch-speech',
                    '-addext', f'subjectAltName={san}'], check=True, capture_output=True)
    os.chmod(key, 0o600)
    return cert, key


def make_handler(hub, send_audio, send_text, max_bytes, log):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype='application/json'):
            if not isinstance(body, bytes):
                body = json.dumps(body).encode()
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ('/', '/index.html'):
                return self._send(200, PAGE.encode(), 'text/html; charset=utf-8')
            if self.path.startswith('/log'):
                try:
                    since = int(self.path.split('since=')[1]) if 'since=' in self.path else -1
                except ValueError:
                    since = -1
                seq, items = hub.since(since, wait=20.0 if since >= 0 else 0.0)  # -1: return at once
                return self._send(200, {'seq': seq, 'items': items})
            self._send(404, {'error': 'not found'})

        def do_POST(self):
            n = int(self.headers.get('Content-Length') or 0)
            if n <= 0 or n > max_bytes:
                self.rfile.read(n) if 0 < n <= 4 * max_bytes else None
                return self._send(413 if n > max_bytes else 400, {'error': f'bad size {n}'})
            body = self.rfile.read(n)
            uid = 'web-' + uuid.uuid4().hex[:6]
            if self.path == '/audio':
                if n % 2:
                    return self._send(400, {'error': 'odd byte count'})
                samples = array.array('h')
                samples.frombytes(body)
                send_audio(uid, samples)
                log(f'[{uid}] phone audio {len(samples) / 16000:.1f} s from {self.client_address[0]}')
                return self._send(200, {'id': uid, 'seconds': round(len(samples) / 16000, 2)})
            if self.path == '/text':
                try:
                    text = json.loads(body).get('text', '').strip()
                except ValueError:
                    text = ''
                if not text:
                    return self._send(400, {'error': 'empty text'})
                send_text(uid, text)
                hub.add({'kind': 'typed', 'id': uid, 'text': text})
                log(f'[{uid}] typed: "{text}"')
                return self._send(200, {'id': uid})
            self._send(404, {'error': 'not found'})

    return Handler


class QuietServer(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        pass  # phones drop connections and reject the self-signed cert on first visit; not worth a traceback


class WebMicNode(Node):
    def __init__(self):
        super().__init__('web_mic')
        p = self.declare_parameter
        port = int(p('port', 8443).value)
        use_https = bool(p('https', True).value)
        cert_dir = p('cert_dir', '~/.config/speech_web').value
        max_seconds = float(p('max_seconds', 15.0).value)

        self.hub = Hub()
        self.audio_pub = self.create_publisher(AudioUtterance, 'speech/utterance', 10)
        self.text_pub = self.create_publisher(Transcript, 'speech/transcript', 10)
        self.create_subscription(Transcript, 'speech/transcript', self._on_transcript, 10)
        self.create_subscription(TaskCommand, 'speech/command', self._on_command, 10)

        handler = make_handler(self.hub, self._send_audio, self._send_text,
                               int(max_seconds * 16000 * 2), self.get_logger().info)
        self.server = QuietServer(('0.0.0.0', port), handler)
        scheme = 'http'
        if use_https:
            cert, key = ensure_cert(cert_dir)
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(cert, key)
            self.server.socket = ctx.wrap_socket(self.server.socket, server_side=True,
                                                 do_handshake_on_connect=False)
            scheme = 'https'
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        urls = ', '.join(f'{scheme}://{ip}:{port}' for ip in local_ips()) or f'{scheme}://<this-pc>:{port}'
        self.get_logger().info(f'Phone page: {urls}' + ('  (accept the certificate warning once)' if use_https else
                                                         '  (http: phones will block the mic; use https)'))

    def _send_audio(self, uid, samples):
        msg = AudioUtterance()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'phone'
        msg.utterance_id = uid
        msg.sample_rate = 16000
        msg.samples = samples
        self.audio_pub.publish(msg)

    def _send_text(self, uid, text):
        msg = Transcript()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.utterance_id = uid
        msg.text = text
        msg.language = 'en'
        msg.confidence = 1.0
        msg.no_speech_prob = 0.0
        self.text_pub.publish(msg)

    def _on_transcript(self, msg):
        if msg.confidence >= 1.0 and msg.utterance_id.startswith('web-'):
            return  # our own typed text, already in the log
        self.hub.add({'kind': 'heard', 'id': msg.utterance_id, 'text': msg.text, 'lang': msg.language,
                      'conf': round(float(msg.confidence), 2), 'uncertain': list(msg.uncertain_words)})

    def _on_command(self, msg):
        self.hub.add({'kind': 'reply', 'id': msg.utterance_id, 'status': STATUS.get(msg.status, str(msg.status)),
                      'task': msg.task, 'args': {kv.key: kv.value for kv in msg.args},
                      'response': msg.response, 'backend': msg.backend, 'conf': round(float(msg.confidence), 2),
                      'transcript': msg.transcript, 'translated_from': _translated_from(msg.raw_json)})


def _translated_from(raw_json):
    try:
        return json.loads(raw_json).get('translated_from')
    except (ValueError, AttributeError):
        return None


def main():
    rclpy.init()
    node = WebMicNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.server.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<meta name="theme-color" content="#F4F2EE">
<title>Talk to Stretch</title>
<style>
:root{
  --bg:#F4F2EE; --panel:#FFFFFF; --ink:#1B1F24; --muted:#6A7079; --line:#DEDAD2;
  --talk:#1E5BB8; --live:#C8372D; --ok:#1C7A47; --ok-soft:#E3F2E9; --ask:#9A6A00; --ask-soft:#FBF0D6;
  --no:#B3372E; --no-soft:#FBE4E1; --off:#7A808A; --off-soft:#ECEDEF;
}
*{box-sizing:border-box;-webkit-tap-highlight-color:transparent}
html,body{height:100%}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.4 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  display:flex;flex-direction:column;max-width:640px;margin:0 auto}
header{display:flex;align-items:center;justify-content:space-between;padding:14px 16px 8px}
header h1{font-size:19px;margin:0;font-weight:700;letter-spacing:-.01em}
header label{font-size:14px;color:var(--muted);display:flex;align-items:center;gap:6px}
#log{flex:1;overflow-y:auto;padding:8px 16px 16px;display:flex;flex-direction:column;gap:10px}
.empty{color:var(--muted);text-align:center;margin:auto 12px;font-size:15px}
.me{align-self:flex-end;max-width:85%;background:var(--talk);color:#fff;border-radius:16px 16px 4px 16px;padding:9px 13px}
.me small{display:block;opacity:.8;font-size:12px;margin-top:2px}
.me.pending{opacity:.6}
.bot{align-self:flex-start;max-width:90%;background:var(--panel);border:1px solid var(--line);border-radius:16px 16px 16px 4px;padding:9px 13px}
.chip{display:inline-block;font-size:12px;font-weight:700;letter-spacing:.04em;text-transform:uppercase;border-radius:999px;padding:1px 8px;margin-bottom:4px}
.execute .chip{background:var(--ok-soft);color:var(--ok)} .clarify .chip{background:var(--ask-soft);color:var(--ask)}
.infeasible .chip{background:var(--no-soft);color:var(--no)} .ignore .chip{background:var(--off-soft);color:var(--off)}
.task{font:13px/1.3 ui-monospace,Menlo,Consolas,monospace;color:var(--muted);margin-top:4px;word-break:break-word}
footer{position:sticky;bottom:0;background:var(--bg);padding:10px 16px calc(14px + env(safe-area-inset-bottom));border-top:1px solid var(--line)}
#talk{width:100%;height:88px;border:0;border-radius:20px;background:var(--talk);color:#fff;font:600 20px/1 inherit;
  touch-action:none;user-select:none;-webkit-user-select:none;display:flex;align-items:center;justify-content:center;gap:12px}
#talk.live{background:var(--live)}
#talk:disabled{opacity:.5}
#talk .dot{width:14px;height:14px;border-radius:50%;background:#fff;opacity:.9}
#talk.live .dot{animation:pulse 1s infinite}
@keyframes pulse{50%{transform:scale(1.5);opacity:.5}}
#meter{height:4px;background:var(--line);border-radius:2px;margin:8px 0 10px;overflow:hidden}
#meter i{display:block;height:100%;width:0;background:var(--live);transition:width .08s}
form{display:flex;gap:8px}
#text{flex:1;min-width:0;border:1px solid var(--line);border-radius:12px;padding:12px;font:inherit;background:var(--panel)}
#send{border:0;border-radius:12px;padding:0 18px;background:var(--ink);color:#fff;font:600 16px inherit}
#status{font-size:13px;color:var(--muted);min-height:18px;margin-top:6px;text-align:center}
:focus-visible{outline:3px solid var(--talk);outline-offset:2px}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#15181C; --panel:#1F2328; --ink:#ECEEF1; --muted:#9AA1AB; --line:#30353C;
  --ok-soft:#173527; --ask-soft:#3A2F12; --no-soft:#3E1D1A; --off-soft:#2A2E34; --ok:#5FC88E; --ask:#E3B54A; --no:#F0817A}}
</style></head><body>
<header><h1>Talk to Stretch</h1>
  <label><input type="checkbox" id="speak"> Read replies aloud</label></header>
<main id="log" aria-live="polite"><p class="empty" id="empty">Hold the button and speak, or type a command.<br>e.g. "let's make a greek salad", "bring me the garlic"</p></main>
<footer>
  <button id="talk" aria-label="Hold to talk"><span class="dot"></span><span id="talkLabel">Hold to talk</span></button>
  <div id="meter"><i></i></div>
  <form id="form"><input id="text" placeholder="…or type a command" autocomplete="off" enterkeyhint="send"><button id="send">Send</button></form>
  <div id="status"></div>
</footer>
<script>
const $ = id => document.getElementById(id);
const status = t => $('status').textContent = t || '';
let ctx = null, stream = null, rate = 48000, recording = false, chunks = [], pre = [], preLen = 0, t0 = 0;
const PRE_S = 0.3, TAIL_MS = 250, MIN_S = 0.35, MAX_S = 15;

try { $('speak').checked = localStorage.getItem('speak') === '1'; } catch (e) {}
$('speak').onchange = () => { try { localStorage.setItem('speak', $('speak').checked ? '1' : '0'); } catch (e) {} };

let wired = false;
async function ensureMic() {
  if (!window.isSecureContext || !navigator.mediaDevices) throw new Error('The microphone needs the https:// address.');
  if (!ctx) { ctx = new (window.AudioContext || window.webkitAudioContext)(); rate = ctx.sampleRate; }
  if (ctx.state === 'suspended') ctx.resume();   // called inside the touch, before any await (iOS)
  if (!stream) stream = await navigator.mediaDevices.getUserMedia(
      {audio: {channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true}});
  if (!wired) {
    wired = true;
    const src = ctx.createMediaStreamSource(stream);
    const proc = ctx.createScriptProcessor(4096, 1, 1);
    proc.onaudioprocess = e => {
      const d = new Float32Array(e.inputBuffer.getChannelData(0));
      let peak = 0; for (let i = 0; i < d.length; i += 16) peak = Math.max(peak, Math.abs(d[i]));
      if (recording) { chunks.push(d); $('meter').firstChild.style.width = Math.min(100, peak * 300) + '%'; }
      else { pre.push(d); preLen += d.length; while (preLen - pre[0].length > PRE_S * rate) preLen -= pre.shift().length; }
    };
    src.connect(proc); proc.connect(ctx.destination);
  }
}

function to16k(parts) {
  let n = 0; for (const p of parts) n += p.length;
  const all = new Float32Array(n); let o = 0; for (const p of parts) { all.set(p, o); o += p.length; }
  const ratio = rate / 16000, m = Math.floor(n / ratio), out = new Int16Array(m);
  for (let i = 0; i < m; i++) {
    const s = Math.floor(i * ratio), e = Math.min(n, Math.floor((i + 1) * ratio));
    let sum = 0; for (let k = s; k < e; k++) sum += all[k];
    out[i] = Math.max(-32768, Math.min(32767, Math.round(sum / Math.max(1, e - s) * 32767)));
  }
  return out;
}

let pressed = false;
async function start(e) {
  e.preventDefault();
  if (recording || pressed) return;
  pressed = true;
  try { talk.setPointerCapture(e.pointerId); } catch (err) {}
  try { await ensureMic(); } catch (err) { pressed = false; status(err.message || 'Microphone not available'); return; }
  if (!pressed) { status('Microphone ready: hold the button while you speak.'); return; }  // released during the permission prompt
  chunks = pre.slice(); pre = []; preLen = 0; recording = true; t0 = performance.now();
  $('talk').classList.add('live'); $('talkLabel').textContent = 'Listening… release to send';
  status('');
  if (navigator.vibrate) navigator.vibrate(15);
  clearTimeout(window._max); window._max = setTimeout(stop, MAX_S * 1000);
}

function stop() {
  pressed = false;
  if (!recording) return;
  clearTimeout(window._max);
  $('talk').classList.remove('live'); $('talkLabel').textContent = 'Hold to talk';
  $('meter').firstChild.style.width = '0';
  const held = (performance.now() - t0) / 1000;   // how long the button was held, without the tail
  setTimeout(async () => {
    recording = false;
    if (held < MIN_S) { status('Too short: hold the button while you speak.'); chunks = []; return; }
    const pcm = to16k(chunks); chunks = [];
    const bubble = addMe('…', 'sending ' + held.toFixed(1) + ' s of audio', true);
    try {
      const r = await fetch('/audio', {method: 'POST', headers: {'Content-Type': 'application/octet-stream'}, body: pcm.buffer});
      const j = await r.json();
      if (!r.ok) throw new Error(j.error || r.status);
      bubble.dataset.id = j.id; bubble.querySelector('small').textContent = 'transcribing…';
    } catch (err) { bubble.querySelector('small').textContent = 'not sent: ' + err.message; }
  }, TAIL_MS);
}

const talk = $('talk');
talk.addEventListener('pointerdown', start);
['pointerup', 'pointercancel', 'pointerleave'].forEach(ev => talk.addEventListener(ev, stop));
talk.addEventListener('contextmenu', e => e.preventDefault());

$('form').onsubmit = async e => {
  e.preventDefault();
  const text = $('text').value.trim(); if (!text) return;
  $('text').value = '';
  try {
    const r = await fetch('/text', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({text})});
    if (!r.ok) throw new Error((await r.json()).error || r.status);
  } catch (err) { status('Not sent: ' + err.message); }
};

function addMe(text, note, pending) {
  $('empty') && $('empty').remove();
  const d = document.createElement('div'); d.className = 'me' + (pending ? ' pending' : '');
  d.innerHTML = '<span></span><small></small>'; d.firstChild.textContent = text; d.lastChild.textContent = note || '';
  $('log').appendChild(d); $('log').scrollTop = 1e9; return d;
}
function addBot(it) {
  $('empty') && $('empty').remove();
  const d = document.createElement('div'); d.className = 'bot ' + it.status;
  const label = {execute: 'Doing it', clarify: 'Question', infeasible: "Can't do", ignore: 'Ignored'}[it.status] || it.status;
  const args = Object.entries(it.args || {}).map(([k, v]) => k + '=' + v).join(', ');
  d.innerHTML = '<span class="chip"></span><div class="say"></div>' + (it.task ? '<div class="task"></div>' : '');
  d.querySelector('.chip').textContent = label;
  d.querySelector('.say').textContent = it.response || (it.status === 'ignore' ? '(not a request)' : '');
  if (it.task) d.querySelector('.task').textContent = it.task + '(' + args + ')  ·  ' + it.backend;
  if (it.translated_from) {
    const t = document.createElement('div'); t.className = 'task';
    t.textContent = 'Understood as: "' + it.transcript + '" (from ' + it.translated_from.language + ')';
    d.insertBefore(t, d.querySelector('.say'));
  }
  $('log').appendChild(d); $('log').scrollTop = 1e9;
  if ($('speak').checked && it.response && window.speechSynthesis) speechSynthesis.speak(new SpeechSynthesisUtterance(it.response));
}

let seq = -1, lost = false;
async function poll() {
  for (;;) {
    try {
      const r = await fetch('/log?since=' + seq); const j = await r.json();
      const first = seq < 0; seq = j.seq;
      if (!first) for (const it of j.items) {
        if (it.kind === 'reply') addBot(it);
        else if (it.kind === 'typed') addMe(it.text, 'typed');
        else if (it.kind === 'heard') {
          const mine = document.querySelector('.me[data-id="' + it.id + '"]');
          const note = 'heard' + (it.lang && it.lang !== 'en' ? ' (' + it.lang + ')' : '') + ' · conf ' + it.conf +
            (it.uncertain && it.uncertain.length ? ' · unsure: ' + it.uncertain.join(', ') : '');
          if (mine) { mine.classList.remove('pending'); mine.firstChild.textContent = it.text || '(nothing heard)'; mine.lastChild.textContent = note; }
          else addMe(it.text, 'robot mic · ' + note);
        }
      }
      if (lost) { status(''); lost = false; }
    } catch (e) { status('Lost connection to the speech PC, retrying…'); lost = true; await new Promise(r => setTimeout(r, 2000)); }
  }
}
poll();
</script></body></html>
"""

if __name__ == '__main__':
    main()
