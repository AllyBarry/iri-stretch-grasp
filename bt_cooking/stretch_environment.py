import json
from typing import Callable, Optional

from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import Trigger

from unaceba.task.environment import Environment

# The task state (beliefs about object locations, target, held object, ...) as JSON.
STATE_TOPIC = "/cooking_bt/state"
# Resets what the robot believes to the priors, e.g. after objects were moved.
RESET_SERVICE = "/cooking_bt/reset_memory"


class StretchEnvironment(Environment):
    '''
    Also publishes the task state on STATE_TOPIC whenever it changes, for monitoring
    what the robot believes (e.g. in the simulator's viewer). The topic is latched, so
    a monitor that starts later still gets the latest state.

    Offers RESET_SERVICE, which calls reset_handler (set by the task). The runner spins
    the node between ticks, so the handler never runs in the middle of a tick.
    '''
    def __init__(self):
        super().__init__()
        self._state_pub = None
        self._last_published = None
        self.reset_handler: Optional[Callable[[], str]] = None

    def inject_ros_node(self, ros_node):
        super().inject_ros_node(ros_node)
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self._state_pub = ros_node.create_publisher(String, STATE_TOPIC, qos)
        ros_node.create_service(Trigger, RESET_SERVICE, self._on_reset)

    def _on_reset(self, request, response):
        if self.reset_handler is None:
            response.success, response.message = False, "this task cannot reset its memory"
        else:
            response.success, response.message = True, self.reset_handler()
        self.log(f"Reset requested on {RESET_SERVICE}: {response.message}")
        return response

    def publish_state(self, tree_name: str, tree, state) -> None:
        if self._state_pub is None:
            return
        tip = tree.tip()
        payload = {
            "tree": tree_name,
            "status": str(tree.root.status.value),
            "tip": tip.name if tip is not None else None,
            "values": state.values,
        }
        text = json.dumps(payload, default=str, sort_keys=True)
        if text == self._last_published:
            return
        self._last_published = text
        self._state_pub.publish(String(data=text))
