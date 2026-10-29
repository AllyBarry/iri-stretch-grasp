# Structured interpretation of a spoken request: ONE high-level task.
# The receiving planner/executor decides how to carry it out (navigation, search, grasping, ...).
uint8 STATUS_EXECUTE=0     # task + args are valid; plan and execute it
uint8 STATUS_CLARIFY=1     # request was unclear/incomplete; response holds the question asked to the user
uint8 STATUS_INFEASIBLE=2  # request understood but the robot cannot do it; response holds the reason
uint8 STATUS_IGNORE=3      # not an actionable request (chit-chat, no wake word, noise)

std_msgs/Header header
string utterance_id
string transcript                  # text that was interpreted
uint8 status
string task                        # e.g. "find" (only meaningful when status == STATUS_EXECUTE, else "")
diagnostic_msgs/KeyValue[] args    # e.g. [{key: "object", value: "keys"}]
string response                    # sentence spoken to the user
float32 confidence                 # 0..1, interpreter confidence
string backend                     # which interpreter produced this (ollama / anthropic / rules / fast_path)
string raw_json                    # full JSON result, for logging/debugging
