# speech_task_interfaces: consuming speech commands

This package defines the messages that the speech pipeline publishes. Your planner or executor needs only this package; it does not need the speech code.

## Quick start

| | |
|---|---|
| **Topic** | `/speech/command` |
| **Type** | `speech_task_interfaces/msg/TaskCommand` |
| **QoS** | default: reliable, volatile, depth 10 (old commands are never replayed to late joiners) |
| **Rate** | one message per user utterance |

Add the package as a dependency of yours:

```xml
<depend>speech_task_interfaces</depend>
```

To watch it live:

```bash
ros2 topic echo /speech/command
ros2 run speech_task text_input      # type commands instead of speaking
```

## TaskCommand
Each message carries **one high-level task**, e.g. `find(object=keys)`, never a sequence of steps.
Breaking the task down (navigation, search, grasping, handover) is your planner's job.

```
uint8 STATUS_EXECUTE=0     # task + args are valid: plan and execute it
uint8 STATUS_CLARIFY=1     # robot is asking the user a question (response); a new message follows the answer
uint8 STATUS_INFEASIBLE=2  # understood but not possible (response = reason)
uint8 STATUS_IGNORE=3      # not a request (noise, chit-chat, a stray "yes")

std_msgs/Header header
string utterance_id                # correlates with /speech/transcript
string transcript                  # what the user said
uint8 status
string task                        # task name, e.g. "find"; "" unless status == EXECUTE
diagnostic_msgs/KeyValue[] args    # e.g. [{key: "object", value: "keys"}]; empty unless EXECUTE
string response                    # sentence spoken to the user (already handled by the speech side)
float32 confidence                 # 0..1
string backend                     # ollama | anthropic | rules | fast_path
string raw_json                    # full interpreter output, for logging
```

**Only `STATUS_EXECUTE` needs action from you.** The speech side handles clarification dialogues and refusals itself.
A `stop` task must preempt whatever is running.

## Tasks
Tasks are defined under `actions:` in `speech_task/config/capabilities.yaml`. Every message is validated against that file before it is published:
- you never receive an unknown task name, or a location that isn't listed;
- optional arguments appear only if the user actually mentioned them, so treat a missing argument as "not specified";
- values are always concrete: pronouns such as "it", "that" or "me" never appear as argument values.

| task | args (✱ = always present) | meaning |
|---|---|---|
| `find` | `object`✱, `location` | locate an object and report where it is (don't pick it up) |
| `fetch` | `object`✱, `from`, `to` | bring an object to the person who asked (or to `to`) |
| `pick_up` | `object`✱, `location` | grasp an object and keep holding it |
| `put` | `location`✱, `object` | put an object (default: the one being held) at a location |
| `give` | none | hand the held object to the person |
| `pass` | `object`✱, `to` | pass an object that is already within reach (e.g. on the table in front of the robot) to a person, without searching for it. `to` = recipient's name; absent = the person who asked |
| `go_to` | `location`✱ | drive to a named location |
| `move` | `direction`✱ (`forward`, `backward`, `turn_left`, `turn_right`), `distance` (0–2, m or rad) | small relative motion |
| `look_at` | `target`✱ | point the head camera |
| `stow` | none | arm to the safe travel pose |
| `say` | `text`✱ | just answer in speech |
| `set_recipe` | `recipe`, `hint` | set the recipe the user is making. **Both args are optional:** `recipe` only if a specific recipe was named; `hint` holds partial information when it wasn't (e.g. "something with chickpeas"). No args = the user wants to cook but hasn't said what |
| `stop` | none | **stop all motion, cancel the current task** |

**Argument values:**
- **Locations** (`location`, `from`, `fetch.to`) are always one of the `vocabulary.locations` list. Currently: `desk`, `table`, `test`, `handover`.
- **Recipients** (`pass.to`) are free text (a person's name), present only when the user named someone else.
- **Objects** are an **open set**: any lowercase name the user says, e.g. `keys`. The `vocabulary.objects` list (currently the ingredients `lemon`, `garlic`, `balsamic`, `tomato`, `onion`, `pepper`, `salt`, `olive oil`, `mustard`, `chickpeas`) improves speech recognition and spelling, but other objects are allowed. If your perception can't handle an object, report that back.
- **Recipes** (`set_recipe.recipe`) are also an open set. `vocabulary.recipes` (currently `hummus`, `vinaigrette`, `tomato salad`, `salsa`, a placeholder list) is a hint. An ingredient or cuisine on its own goes in `hint`, never in `recipe`.
- **Hints** (`set_recipe.hint`) are free text.

If you add or rename a task on your side, update `capabilities.yaml` to match.

## Examples

| User says | `status` | `task` | `args` |
|---|---|---|---|
| "where are my keys" | EXECUTE | `find` | `object=keys` |
| "bring me the apple from the table" | EXECUTE | `fetch` | `object=apple`, `from=table` |
| "put it on the desk" | EXECUTE | `put` | `location=desk` |
| "pass me the mustard" | EXECUTE | `pass` | `object=mustard` |
| "could you pass the apple to Anna" | EXECUTE | `pass` | `object=apple`, `to=Anna` |
| "pass it over" | CLARIFY | — | — ("What should I pass?") |
| "go to the kitchen" | CLARIFY | — | — ("Did you mean one of: desk, table, …?") |
| "make me a sandwich" | INFEASIBLE | — | — |
| "let's make hummus" | EXECUTE | `set_recipe` | `recipe=hummus` |
| "we're making shakshuka tonight" | EXECUTE | `set_recipe` | `recipe=shakshuka` |
| "let's make something with chickpeas" | EXECUTE | `set_recipe` | `hint=something with chickpeas` |
| "can you set the recipe" | EXECUTE | `set_recipe` | — (recipe unknown) |
| "stop!" | EXECUTE | `stop` | — |

The first example as published:

```yaml
status: 0                       # EXECUTE
transcript: where are my keys
task: find
args:
  - {key: object, value: keys}
response: Let me check where your keys might be.
```

## Minimal consumer

**Python**

```python
import rclpy
from rclpy.node import Node
from speech_task_interfaces.msg import TaskCommand


class Executor(Node):
    def __init__(self):
        super().__init__('task_executor')
        self.create_subscription(TaskCommand, '/speech/command', self.on_command, 10)

    def on_command(self, msg: TaskCommand):
        if msg.status != TaskCommand.STATUS_EXECUTE:
            return
        args = {kv.key: kv.value for kv in msg.args}
        if msg.task == 'stop':
            self.cancel_everything()
            return
        self.get_logger().info(f'{msg.task} {args}')
        # plan + execute msg.task with args, e.g. find -> search for args['object'] ...

    def cancel_everything(self):
        pass


rclpy.init()
rclpy.spin(Executor())
```

**C++**

```cpp
#include "speech_task_interfaces/msg/task_command.hpp"
using speech_task_interfaces::msg::TaskCommand;

sub_ = create_subscription<TaskCommand>("/speech/command", 10,
  [this](TaskCommand::ConstSharedPtr msg) {
    if (msg->status != TaskCommand::STATUS_EXECUTE) return;
    for (const auto & kv : msg->args)
      RCLCPP_INFO(get_logger(), "%s %s=%s", msg->task.c_str(), kv.key.c_str(), kv.value.c_str());
  });
```

## Testing without speech

```bash
ros2 service call /interpreter/parse_text speech_task_interfaces/srv/ParseText \
  "{text: 'where are my keys', use_dialogue_context: false}"

ros2 topic pub --once /speech/transcript speech_task_interfaces/msg/Transcript \
  "{text: 'bring me the apple from the table', confidence: 1.0}"   # result appears on /speech/command
```

## Other messages in this package
You don't need these to consume commands.

| Type | Topic | Purpose |
|---|---|---|
| `AudioUtterance` | `/speech/utterance` | 16 kHz int16 audio of one speech segment |
| `std_msgs/Bool` | `/speech/ptt` | push-to-talk key state: true while the talk key is held (only in push-to-talk mode) |
| `Transcript` | `/speech/transcript` | speech-to-text output: `text`, `confidence`, `no_speech_prob`, `uncertain_words` (possible mis-hearings) |
| `ParseText` (srv) | `/interpreter/parse_text` | text in → `TaskCommand` out |

## Running on another machine
Use the same `ROS_DOMAIN_ID` on both machines (the speech workspace defaults to `42`) and build this package on both.
