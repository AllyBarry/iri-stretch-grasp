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

        for obj in sorted(self.get_objects()):
            # Locations are mutually exclusive, so probabilities add.
            prob = sum(state.get_value(f"confidence/{obj}/{loc}") for loc in self._ACCESSIBLE_LOCATIONS)
            prob = min(prob, 1.0)  # guard against rounding / non-normalised data

            state.set_value(f"reachable/{obj}",prob > self.reachable_threshold)
            state.set_value(f"confidence/reachable/{obj}",prob)
            state.set_value(f"required/{obj}",obj in required)
            state.set_value(f"confidence/required/{obj}",1.0) # TODO: What about underspecified recipes?

        return py_trees.common.Status.SUCCESS

    def input_variables(self):
        location_confs = [f"confidence/{obj}/{loc}" for obj in self.get_objects() for loc in self._ACCESSIBLE_LOCATIONS]


        return location_confs + ["recipe/name"]

    def output_variables(self):
        reachables = [f"reachable/{obj}" for obj in self.get_objects()]
        reachable_confs = [f"confidence/reachable/{obj}" for obj in self.get_objects()]
        requireds = [f"required/{obj}" for obj in self.get_objects()]
        required_confs = [f"confidence/required/{obj}" for obj in self.get_objects()]

        return reachables + reachable_confs + requireds + required_confs


    def get_objects(self):
        return set(StretchState._INGREDIENTS) | set(StretchState._TOOLS)

'''
TEST
'''
class TestNode(ActionNode):
    def __init__(self, name: str = "TestNode"):
        super().__init__(name)

        self.board.register_key("environment", access=py_trees.common.Access.READ)

    def execute(self, state: StretchState):
        state.set_value("recipe/name","caprese_salad")

        non_reachable = [
            key[len("reachable"):]
            for key, value in state.values.items()
            if key.startswith("reachable") and value is False
        ]
        print(non_reachable)

        return py_trees.common.Status.SUCCESS

    def input_variables(self):
        return []

    def output_variables(self):
        return []

    