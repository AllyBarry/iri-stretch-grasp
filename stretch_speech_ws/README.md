# stretch_speech_ws: speech → structured high-level tasks (ROS 2)

Turns a spoken request into **one validated high-level task** for the Stretch RE2, or into a
**clarification question**, an **"I can't do that" with a reason**, or **ignore**.
A separate planner/executor receives the task and decides how to carry it out.

```
 "where are my keys"                      "bring me the apple from the table"
        │                                          │
        ▼                                          ▼
 TaskCommand status=EXECUTE                TaskCommand status=EXECUTE
   task=find  args={object: keys}            task=fetch  args={object: apple, from: table}
```

Consumers only need `speech_task_interfaces`; see [its README](src/speech_task_interfaces/README.md) for the message format and task list.

## Architecture

```
            ROBOT (Stretch RE2, NUC, no GPU)          │          OFFBOARD PC (RTX GPU)
                                                      │
 ReSpeaker ─► audio_capture ── /speech/utterance ─────┼──► transcriber (faster-whisper, CUDA)
 mic array    (VAD or PTT)      AudioUtterance        │         │ /speech/transcript (Transcript)
                ▲                                     │         ▼
                │ /speech/speaking (Bool, mute)       │    interpreter ──► LLM backend
 speaker ◄── tts ◄────────── /speech/command ◄────────┼──── (fast paths,   ├ ollama   (local GPU, default)
                                TaskCommand           │     validation,    ├ anthropic (Claude API)
 planner/executor ◄────────── /speech/command ◄───────┘     dialogue)      └ rules    (fallback, no deps)
 (separate service: one task in,                                  ▲
  Nav2, grasping, ... out)                                        │
                                                 config/capabilities.yaml
```

### Recommended deployment
**Split: capture, TTS and execution run on the robot; ASR and the LLM run on this PC.**

| Component | Where | Why |
|---|---|---|
| `audio_capture` | robot | The mic is on the robot. VAD is cheap, and only finished utterances (~32 kB/s of speech) cross Wi-Fi, never a continuous stream. |
| `transcriber` | PC | Whisper (`distil-large-v3`, ~0.3 s per utterance here) needs a GPU. On the robot's NUC CPU it's several times slower, and larger models aren't practical. |
| `interpreter` | PC | A 7–8B local LLM needs about 5 GB of VRAM. The robot can't run one. The Claude backend works from either machine but needs internet. |
| `tts`, executor | robot | The speaker is on the robot. The executor must keep working, and stop safely, if the PC or network drops. |

For development, everything runs on this PC with its own mic (the current default).

**Splitting across machines.** Stretch's `stretch_ros2` targets **ROS 2 Humble / Ubuntu 22.04**, and this PC runs **Jazzy**.
Mixing distros over DDS often works for simple messages, but it is not officially supported. Options, most robust first:
1. Run the PC side in a **Humble Docker container** (`--net=host`, `--gpus all`). Then both ends share the distro and message definitions.
2. Build `speech_task_interfaces` on both machines with the same `ROS_DOMAIN_ID` and test cross-distro discovery.
3. Bridge the distros with `zenoh-bridge-ros2dds`, which also handles lossy Wi-Fi better.

Either way, `speech_task_interfaces` has to be built on the robot too. It depends only on `std_msgs` and `diagnostic_msgs`.

**Safety.** The interpreter answers "stop / halt / freeze …" on a regex fast path that skips the LLM queue. That fast path still runs on the PC, though.
The robot-side executor should keep its own e-stop, and should cancel motion if `/speech/command` or the PC goes silent.

## How a request is decided
The interpreter (`speech_task/interpreter.py`) applies these steps in order:
1. Empty text, or Whisper `no_speech_prob` is high → **IGNORE**
2. The utterance is a stop phrase → **EXECUTE [stop]** immediately, without calling the LLM
3. ASR confidence is low → **CLARIFY** "Sorry, I didn't catch that…" (no LLM call)
4. Wake words are configured and none was heard → **IGNORE**
5. A bare "yes", "no", "okay" or "thanks" while no question is open → **IGNORE** (it isn't a request)
6. The LLM is called with a system prompt generated from `capabilities.yaml`, plus any words Whisper flagged as uncertain (see below). It must return **one task**, never a sequence of steps. Its output is constrained by a JSON schema, so task names can only come from the manifest. A request containing several separate tasks gets a CLARIFY asking which to do first.
7. The **validator** enforces the manifest:
   - unknown task → INFEASIBLE;
   - missing required argument → CLARIFY, using the param's `ask` text;
   - a location not in the list → CLARIFY, offering the valid options;
   - number out of range → INFEASIBLE.

   Objects are an open set (`strict: false`), so unlisted objects such as "keys" are accepted.
   Values that name nothing concrete ("it", "that", "me", …) count as not given. So "pass it over" asks "What should I pass?", and "pass me the cup" doesn't send `to=me`.
8. **Grounding check.** A location or object argument must have been said in this conversation: by the user, as a sound-alike ("mustered" → mustard), or in the robot's own question that the user is answering. This stops small LLMs from inventing a location nobody mentioned.
   - If the argument is optional, it is dropped, and the spoken reply is regenerated.
   - If it is required, the result becomes a confirmation question.

   A synonym the user did say (e.g. "lounge" for `living_room`) also counts as not mentioned, so a required one gets a confirmation question.
9. An EXECUTE result with confidence below `min_plan_confidence` becomes a confirmation question ("Just to check: should I …?").
   An EXECUTE reply never asks anything: if the LLM's acknowledgement contains a question, it is replaced with "Okay, I'll …". Questions are only for CLARIFY, which is the only status that routes the user's answer back.
10. **Dialogue:** after a CLARIFY, the next utterance (within 30 s) is sent to the LLM together with the original request and the question; a "yes" confirms the robot's suggestion. If 3 rounds pass without a task, the interpreter resets and asks "Let's start over: what would you like me to do?". That reset is a CLARIFY, not INFEASIBLE, because nothing was judged impossible.

Whenever the validator or grounding check overrides the LLM, the reason is logged as `[validator] …`, e.g. `execute -> clarify: go_to.location='kitchen' not in locations`, and stored as `validation_note` in `raw_json`.

If the LLM backend is unreachable or errors, the `rules` backend answers instead.

## Speech recognition (transcription)
The `transcriber` node runs [faster-whisper](https://github.com/SYSTRAN/faster-whisper) on the GPU. Three mechanisms keep mis-hearings from turning into wrong or needless questions.

**1. Priming with example commands (`asr.phrases` in `capabilities.yaml`).**
Whisper favours words, and wording, that appear in its prompt. The node builds that prompt from template sentences, filling `{object}` and `{location}` in turn from the `vocabulary:` lists:

```yaml
asr:
  phrases:
    - "Fetch the {object}."
    - "Bring me the {object} from the {location}."
    - "Go to the {location}."
  uncertain_word_prob: 0.5
```

This produces a prompt like "Stretch, Fetch the apple. Bring me the mustard from the desk. Go to the table. … Stop." Any vocabulary word not used by a template is appended, so every place and object appears at least once. The node logs the prompt at startup.

**Include the verbs, not just the nouns.** When only nouns were listed, "fetch my keys" was transcribed as "fitch my keys". Whenever a word is misheard repeatedly, add a phrase containing it. Keep the list to about 20 short phrases; very long prompts can make Whisper repeat the prompt back.

**2. Uncertain words are passed to the interpreter.** Words whose Whisper probability is below `uncertain_word_prob` go into `Transcript.uncertain_words`. The interpreter appends them to the LLM request as "possibly misheard". The LLM is told to correct them silently to the closest-sounding listed word or command verb, and to ask only if two corrections both make sense.

Tested with the local `qwen2.5:7b-instruct`:

| Transcript (flagged) | Result |
|---|---|
| "fitch the apple" (`fitch`) | `fetch(object=apple)` |
| "fitch the mustered from the desk" (`fitch`, `mustered`) | `fetch(object=mustard, from=desk)` |
| "ring me the pringles" (`ring`) | `fetch(object=pringles)` |

The transcript-level `confidence` is still used separately. If the whole utterance is low-confidence, the robot asks the user to repeat it, without calling the LLM.

**3. Model choice.** Set it with `whisper_model:=` at launch. Times are measured on this PC's RTX 3070 Ti Laptop GPU (8 GB):

| Model | ~Time per utterance | Notes |
|---|---|---|
| `small.en` | 0.1 s | fastest, least accurate |
| `medium.en` | 0.2 s | |
| `distil-large-v3` | 0.3–0.4 s | **default**, English |
| `large-v3-turbo` | 0.4–0.7 s | multilingual |

On a GPU, Whisper runs as `int8_float16`. This saves about 1 GB, which is what lets it share 8 GB with the 7B LLM: about 6.7 GB in use, versus 7.7 GB with `float16`. Override with `-p compute_type:=float16`.

**Microphone matters as much as the model.** On the robot use the ReSpeaker (`mic_device:=ReSpeaker`): its processed channel has beamforming and noise suppression. Speak within a couple of metres, and avoid running TTS into the mic (the `tts` node mutes capture while speaking).

### Measuring accuracy on your own voice
Change one thing at a time and measure word error rate (WER) on real recordings:

```bash
source ~/iri-stretch-grasp/stretch_speech_ws/env.sh
# 1. record: every detected utterance is also saved as <id>.wav
ros2 run speech_task audio_capture --ros-args -p save_dir:=$HOME/asr_data
# 2. label: create ~/asr_data/transcripts.tsv, one line per file:  <id>.wav<TAB>what you actually said
# 3. compare models, each with and without the prompt
cd ~/iri-stretch-grasp/stretch_speech_ws/src/speech_task
python3 -m speech_task.asr_bench ~/asr_data --models small.en distil-large-v3 medium.en
```

`asr_bench` prints WER and milliseconds per utterance for each model/prompt combination. It also lists every wrong transcription next to what was actually said; anything that keeps going wrong belongs in `asr.phrases`. About 20–30 utterances, from the people who will actually use the robot, is enough to compare options.

If mis-hearings still get through, use a stronger LLM (`backend:=anthropic`, or a newer local model such as `qwen3:8b`). It repairs transcripts better from context.

## Listening modes
`audio_capture` has two ways of deciding when the user is talking. Choose with `listen_mode:=` at launch, or `-p mode:=` on the node.

| Mode | Behaviour | Good for |
|---|---|---|
| `vad` (default) | Always listening. Voice activity detection starts an utterance on speech and ends it after 0.7 s of silence. | hands-free use |
| `push_to_talk` | Records **only while SPACE is held** in the push-to-talk window. Release to send. | noisy rooms, demos, labs where people talk near the robot |

```bash
ros2 launch speech_task speech_pipeline.launch.py listen_mode:=push_to_talk
```

This opens a small **"Stretch push-to-talk"** window:
- Click it once so it has keyboard focus, then **hold SPACE** while you speak. Holding the left mouse button on the big button works too.
- The window shows what was heard and the robot's interpretation: status, reply and task.
- Keys only count while the window is focused, so typing spaces in other programs never opens the mic.
- To use a different key: `ros2 run speech_task push_to_talk --ros-args -p key:=F1` (any Tk key name).

How it works:
- **Key state as a topic.** The window publishes `/speech/ptt` (`std_msgs/Bool`): `true` while the key is held, re-sent every 200 ms as a heartbeat, and `false` on release. Anything else can drive push-to-talk by publishing the same topic, such as a gamepad button or a GUI.
- **Recording window.** `audio_capture` records while `/speech/ptt` is true. It adds 0.3 s from before the press (`pre_roll_ms`) and 0.25 s after release (`release_tail_ms`), so the first and last words aren't clipped.
- **Accidental taps.** A press shorter than `min_utterance_ms` (0.4 s) is discarded.
- **Stuck-key safety.** If the heartbeat stops for `ptt_timeout_s` (1 s), e.g. because the window crashed or the network dropped, the key is treated as released. The window also releases when it loses focus, and recordings are capped at `max_utterance_ms` (15 s).
- **TTS overlap.** While the robot is speaking, mic frames are skipped, but the utterance being recorded is kept.

**Split deployment.** The key and the mic can be on different machines:

```bash
# robot (mic):        ros2 launch speech_task speech_pipeline.launch.py listen_mode:=push_to_talk run_ptt_ui:=false run_compute:=false mic_device:=ReSpeaker
# PC (keyboard, GPU): ros2 launch speech_task speech_pipeline.launch.py listen_mode:=push_to_talk run_audio:=false run_tts:=false
```

The window needs a desktop session (an X11 or Wayland `DISPLAY`). Run it from a terminal on the PC's desktop, not over plain SSH.

## Interfaces (`speech_task_interfaces`)
| Topic / service | Type | |
|---|---|---|
| `/speech/utterance` | `AudioUtterance` | 16 kHz int16 mono, one per VAD segment |
| `/speech/transcript` | `Transcript` | text, confidence, no_speech_prob, uncertain_words |
| `/speech/command` | `TaskCommand` | `status` ∈ EXECUTE / CLARIFY / INFEASIBLE / IGNORE, `task` + `args` (one high-level task), `response` (what to say), `raw_json` |
| `/speech/speaking` | `std_msgs/Bool` | the TTS node mutes the mic while the robot is talking |
| `/speech/ptt` | `std_msgs/Bool` | push-to-talk key state (`listen_mode:=push_to_talk`): true while held |
| `/interpreter/parse_text` | `ParseText.srv` | text in → `TaskCommand` out (no audio) |

## Setup (this PC)
This is a colcon workspace of its own, in the `stretch_speech_ws/` folder of the
`iri-stretch-grasp` repository. Build it here, not in the robot's workspace:

```bash
git clone git@github.com:AllyBarry/iri-stretch-grasp.git ~/iri-stretch-grasp
cd ~/iri-stretch-grasp/stretch_speech_ws
/usr/bin/python3 -m venv --system-site-packages .venv   # system Python (ROS Jazzy's), not conda
.venv/bin/pip install -r src/speech_task/requirements.txt
source env.sh                                     # deactivates conda, sources ROS + venv + workspace
colcon build --symlink-install && source env.sh
```
`env.sh` is required because conda's Python breaks `rclpy` (GLIBCXX error).

The venv and the colcon `build/` and `install/` folders store absolute paths, so if the
workspace moves, delete `.venv`, `build`, `install` and `log` and repeat the steps above.

**LLM backend (pick one):**
- **Ollama, local (default):**
  `curl -fsSL https://ollama.com/install.sh | sh && ollama pull qwen2.5:7b-instruct`
  This model plus Whisper `distil-large-v3` fits in 8 GB of VRAM. For lower latency use a 3B model:
  `model:=qwen2.5:3b-instruct`.
- **Claude:** `export ANTHROPIC_API_KEY=...` (or `ant auth login`), then launch with `backend:=anthropic`.
  It defaults to `claude-opus-5-5` at `effort: low`, to keep voice turn-around short.
  Server-side refusal fallback is enabled.
- **None:** `backend:=rules`. Keyword matching only; good for plumbing tests.

TTS is optional: `sudo apt install espeak-ng`. Without it, responses are only logged.

## Run
```bash
# full pipeline on this PC (mic → command); add listen_mode:=push_to_talk to talk only while SPACE is held
ros2 launch speech_task speech_pipeline.launch.py                      # backend:=anthropic / rules, whisper_model:=small.en, mic_device:=ReSpeaker

# no mic: type commands (in a second terminal, while the launch runs with run_audio:=false)
ros2 run speech_task text_input

# no ROS at all: try the interpreter directly
cd src/speech_task && python3 -m speech_task.cli --backend ollama -v
python3 -m speech_task.cli --backend rules "go somewhere" "the desk"

# test the transcriber with a recording
ros2 run speech_task wav_publisher --ros-args -p file:=/path/to/16bit.wav

# split deployment
#   robot: ros2 launch speech_task speech_pipeline.launch.py run_compute:=false mic_device:=ReSpeaker
#   PC:    ros2 launch speech_task speech_pipeline.launch.py run_audio:=false run_tts:=false
#   both machines need the same ROS_DOMAIN_ID (env.sh defaults to 42)
```

Tests: `cd src/speech_task && python3 -m pytest test/`

## Adapting it
- **Add or remove tasks:** edit `config/capabilities.yaml` (`actions:` = tasks, params, vocabulary, limitations, stop phrases, wake words).
  Set `strict: false` on a parameter to accept values outside its vocabulary list (as for objects).
  The LLM prompt, the JSON schema, the validator, and the Whisper prompt (`asr.phrases`) all regenerate from this file.
  Keep it in sync with what your planner supports.
- **Planner/executor (separate service):** subscribe to `/speech/command` and act only on `STATUS_EXECUTE`.
  Decompose each task into Stretch behaviours. For example, `fetch` → Nav2 to `from` (or search) → grasp → return to the person → hand over.
  It should report failures back, for example by publishing a `TaskCommand` with `say`.
- **Grounding:** the location and object lists are static for now. Feeding the interpreter what the robot currently perceives or knows would let it answer "infeasible: I can't see a cup here" or ask "which cup?".
