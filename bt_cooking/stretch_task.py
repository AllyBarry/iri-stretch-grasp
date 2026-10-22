import py_trees

from unaceba.task.ros_task import RosTask

from bt_cooking.stretch_state import StretchState
from bt_cooking.stretch_environment import StretchEnvironment
from bt_cooking.stretch_bt import test_sequence

class StretchTask(RosTask):
    def __init__(self, log_dir, log_file, **kwargs):
        super().__init__(log_dir, log_file, **kwargs)

        # TODO: Implement setup

    # ------------------------------------------------------------------ #
    #  Task setup                                                        #
    # ------------------------------------------------------------------ #
    def setup_state(self, **kwargs):
        return StretchState()

    def setup_environment(self, **kwargs):
        return StretchEnvironment()

    def setup_tree(self, **kwargs):
        return py_trees.trees.BehaviourTree(root=test_sequence())

    
    # ------------------------------------------------------------------ #
    #  Injected Functions                                                #
    # ------------------------------------------------------------------ #

    def tree_done_condition(self, ros_node):
        return True
    