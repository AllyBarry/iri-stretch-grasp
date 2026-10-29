"""Compare Whisper models / prompting on your own recordings.

  1. record:  ros2 run speech_task audio_capture --ros-args -p save_dir:=~/asr_data
  2. label:   ~/asr_data/transcripts.tsv with lines "<file>.wav<TAB>what you actually said"
  3. run:     python3 -m speech_task.asr_bench ~/asr_data --models small.en distil-large-v3

Prints word error rate (WER) per model, with and without the capabilities.yaml prompt,
plus every wrong transcription so you can see what to add to `asr.phrases`.
"""
import argparse
import glob
import os
import re
import time
import wave

import numpy as np

from .capabilities import Capabilities

DEFAULT_CAPS = os.path.join(os.path.dirname(__file__), '..', 'config', 'capabilities.yaml')


def words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]", ' ', text.lower()).split()


def wer(ref: str, hyp: str) -> tuple[int, int]:
    """(edit distance in words, number of reference words)"""
    r, h = words(ref), words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
    return d[len(h)], len(r)


def load(path: str) -> np.ndarray:
    with wave.open(path) as w:
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).reshape(-1, w.getnchannels())[:, 0]
        assert w.getframerate() == 16000, f'{path}: need 16 kHz'
    return a.astype(np.float32) / 32768.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dir')
    ap.add_argument('--models', nargs='+', default=['small.en', 'distil-large-v3'])
    ap.add_argument('--caps', default=DEFAULT_CAPS)
    ap.add_argument('--compute_type', default='float16')
    a = ap.parse_args()

    from faster_whisper import WhisperModel
    refs = {}
    tsv = os.path.join(a.dir, 'transcripts.tsv')
    if os.path.exists(tsv):
        for line in open(tsv):
            if '\t' in line:
                f, t = line.rstrip('\n').split('\t', 1)
                refs[f] = t
    files = sorted(glob.glob(os.path.join(a.dir, '*.wav')))
    if not files:
        raise SystemExit(f'no .wav files in {a.dir}')
    prompt = Capabilities.from_yaml(a.caps).asr_prompt()

    for name in a.models:
        model = WhisperModel(name, device='cuda', compute_type=a.compute_type)
        for label, p in (('no prompt', None), ('prompt', prompt)):
            errs = total = 0
            t0 = time.monotonic()
            print(f'\n=== {name} / {label}')
            for f in files:
                segs, _ = model.transcribe(load(f), language='en', beam_size=5, vad_filter=True,
                                           initial_prompt=p, condition_on_previous_text=False)
                hyp = ' '.join(s.text.strip() for s in segs)
                ref = refs.get(os.path.basename(f))
                if ref is None:
                    print(f'  {os.path.basename(f)}: {hyp}')
                    continue
                e, n = wer(ref, hyp)
                errs, total = errs + e, total + n
                if e:
                    print(f'  {os.path.basename(f)}: "{hyp}"  (said: "{ref}")')
            dt = (time.monotonic() - t0) / len(files)
            print(f'  -> WER {100 * errs / total:.1f}%' if total else '  -> (no references)', f'| {dt * 1000:.0f} ms/utt')
        del model


if __name__ == '__main__':
    main()
