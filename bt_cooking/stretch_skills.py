import py_trees

from unaceba.bt.nodes import ActionNode,ConditionNode
from unaceba.bt.rosnodes import RosActionNode

from bt_cooking.stretch_state import StretchState

'''
DECISION-MAKING
'''
class CalculateReachabilityNode(ActionNode):
    _ACCESSIBLE_LOCATIONS = ["Fridge","Table"]

    def __init__(self, name:str = "CalculateReachabilityNode", reachable_threshold: float = 0.5):
        super(CalculateReachabilityNode, self).__init__(name)

        self.reachable_threshold = reachable_threshold
        self.board.register_key("environment", access=py_trees.common.Access.WRITE)

    def execute(self, state:StretchState):
        current_recipe = state.get_value("recipe/name")
        recipe_data = StretchState.load_recipe(current_recipe)

        required = set(recipe_data.get("ingredients", [])) | set(recipe_data.get("tools", []))

        for obj in sorted(StretchState.get_objects()):
            # Locations are mutually exclusive, so probabilities add.
            prob = sum(state.get_value(f"confidence/{obj}/{loc}") for loc in self._ACCESSIBLE_LOCATIONS)
            prob = min(prob, 1.0)  # guard against rounding / non-normalised data

            state.set_value(f"human_reachable/{obj}",prob > self.reachable_threshold)
            state.set_value(f"confidence/human_reachable/{obj}",prob)
            state.set_value(f"required/{obj}",obj in required)
            state.set_value(f"confidence/required/{obj}",1.0) # TODO: What about underspecified recipes?

        return py_trees.common.Status.SUCCESS

    def input_variables(self):
        location_confs = [f"confidence/{obj}/{loc}" for obj in StretchState.get_objects() for loc in self._ACCESSIBLE_LOCATIONS]


        return location_confs + ["recipe/name"]

    def output_variables(self):
        objects = StretchState.get_objects()

        reachables = [f"human_reachable/{obj}" for obj in objects]
        reachable_confs = [f"confidence/human_reachable/{obj}" for obj in objects]
        requireds = [f"required/{obj}" for obj in objects]
        required_confs = [f"confidence/required/{obj}" for obj in objects]

        return reachables + reachable_confs + requireds + required_confs

    def semantic_description(self):
        return "Calculates the human reachability of each object and whether or not they are required by the recipe known to the robot"

class DecideTargetObjectNode(ActionNode):
    def __init__(self, name:str = "DecideTargetObjectNode"):
        super(DecideTargetObjectNode, self).__init__(name)

        self.board.register_key("environment", access=py_trees.common.Access.WRITE)

    def execute(self, state:StretchState):
        current_recipe = state.get_value("recipe/name")
        recipe_data = StretchState.load_recipe(current_recipe)

        ordered_objects = self.get_objects_by_first_appearance(recipe_data)

        blocking_objects = [obj for obj in StretchState.get_objects() if state.get_value(f"required/{obj}") and not state.get_value(f"human_reachable/{obj}")]

        blocking = set(blocking_objects)
        next_obj = next((obj for obj in ordered_objects if obj in blocking), None)

        state.set_value("target_object",next_obj)

        return py_trees.common.Status.SUCCESS

    
    def input_variables(self):
        known_objects = StretchState.get_objects()
        requireds = [f"required/{obj}" for obj in known_objects]
        reachables = [f"human_reachable/{obj}" for obj in known_objects]
        return requireds + reachables + ["recipe/name"] 

    def output_variables(self):
        return ["target_object"]

    def semantic_description(self):
        return "Based on which objects are reachable by the human, which are required by the recipe, and the order of steps in the recipe, decides the next object to fetch"


    @staticmethod
    def get_objects_by_first_appearance(recipe_data):
        known_objects = StretchState.get_objects()
        arg_keys = ("object", "tool", "source", "target")

        seen = set()
        ordered = []

        for step in recipe_data.get("steps", []):
            for action in step.get("actions", []):
                args = action.get("args", {})
                for key in arg_keys:
                    value = args.get(key)
                    if value in known_objects and value not in seen:
                        seen.add(value)
                        ordered.append(value)

        return ordered

'''
EXECUTING ROBOT SKILLS
'''
# TODO: Navigate to location
class NavigateToPlaceNode(RosActionNode):
    SERVICE_NAME:   str   = "TODO"
    SERVER_TIMEOUT: float = 5.0

    def __init__(self, name: str = "NavigateToPlaceNode"):
        super().__init__(name)
        self._client         = None
        self._response_future = None
        self._sent           = False

        self.board.register_key("environment", access=py_trees.common.Access.READ) 

    def input_variables(self) -> List[str]:
        return ["target_location"]

    def output_variables(self) -> List[str]:
        return []

    def ros_setup(self) -> None:
        self._client = self._ros_node.create_client(
            TODO_MESSAGE_TYPE,
            self.SERVICE_NAME
        )

    def initialise(self) -> None:
        self._response_future = None
        self._sent            = False

        self._ros_node.get_logger().info(
            f"[{self.name}] Waiting for service '{self.SERVICE_NAME}'..."
        )

        if not self._client.wait_for_service(timeout_sec=self.SERVER_TIMEOUT):
            self._ros_node.get_logger().error(
                f"[{self.name}] Service '{self.SERVICE_NAME}' not available "
                f"after {self.SERVER_TIMEOUT}s."
            )
            return  # _response_future stays None; execute() returns FAILURE

        state         = self.board.state
        target_location = state.get_value("target_location")

        request              = TODO_MESSAGE_TYPE.Request()
        request.target_location  = target_location if target_location is not None else "" # TODO: Fix to conform to message type

        self._response_future = self._client.call_async(request)
        self._sent            = True

    def execute(self, state) -> py_trees.common.Status:
        # Service was unreachable
        if self._response_future is None:
            return py_trees.common.Status.FAILURE

        # Waiting on operator
        if not self._response_future.done():
            return py_trees.common.Status.RUNNING

        response = self._response_future.result()

        if response is None:
            self._ros_node.get_logger().error(
                f"[{self.name}] Service call returned no response."
            )
            return py_trees.common.Status.FAILURE

        if response.success:
            self._ros_node.get_logger().info(
                f"[{self.name}] Navigation successful. "
                f"Message: '{response.message}'"
            )
            return py_trees.common.Status.SUCCESS
        else:
            self._ros_node.get_logger().warn(
                f"[{self.name}] Navigation failed. "
                f"Message: '{response.message}'"
            )
            return py_trees.common.Status.FAILURE

    def terminate(self, new_status) -> None:
        if self._sent and self._response_future is not None \
                and not self._response_future.done():
            self._ros_node.get_logger().warn(
                f"[{self.name}] Preempted before navigation finished "
                f"(terminated with {new_status}). Response will be discarded."
            )

        self._response_future = None
        self._sent            = False

# TODO: Grasp object
# TODO: Confirm grasp
# TODO: Handover

'''
TEST
'''
class TestNode(ActionNode):
    def __init__(self, name: str = "TestNode"):
        super().__init__(name)

        self.board.register_key("environment", access=py_trees.common.Access.READ)

    def execute(self, state: StretchState):
        state.set_value("recipe/name","caprese_salad")

        return py_trees.common.Status.SUCCESS

    def input_variables(self):
        return []

    def output_variables(self):
        return []

    