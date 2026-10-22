import py_trees

from unaceba.bt.nodes import ActionNode,ConditionNode
from unaceba.bt.rosnodes import RosActionNode

from bt_cooking.stretch_state import StretchState



'''
TEST
'''
class TestNode(ActionNode):
    def __init__(self, name: str = "TestNode"):
        super().__init__(name)

        self.board.register_key("environment", access=py_trees.common.Access.READ)

    def execute(self, state: StretchState):
        return py_trees.common.Status.SUCCESS

    def input_variables(self):
        return []

    def output_variables(self):
        return []

    