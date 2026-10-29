import inspect
from typing import Dict, Callable, Optional

import py_trees

from unaceba.task.ros_task import RosTask

from bt_cooking import stretch_bt
from bt_cooking.stretch_state import StretchState
from bt_cooking.stretch_environment import StretchEnvironment

Status = py_trees.common.Status

DEFAULT_TREE = "test_sequence"


def available_trees() -> Dict[str, Callable[[], py_trees.behaviour.Behaviour]]:
    '''The tree builders defined in bt_cooking.stretch_bt, by function name.'''
    return {
        name: fn for name, fn in inspect.getmembers(stretch_bt, inspect.isfunction)
        if fn.__module__ == stretch_bt.__name__ and not name.startswith("_")
    }


class StretchTask(RosTask):
    '''
    kwargs:
        tree:       name of a tree builder in bt_cooking.stretch_bt (default test_sequence)
        recipe:     recipe id to set as "recipe/name" before the first tick, e.g. greek_salad
        max_passes: when looping, stop after the tree has finished this many times
    '''
    def __init__(self, log_dir, log_file, tree: str = DEFAULT_TREE, recipe: Optional[str] = None,
                 max_passes: int = 10, **kwargs):
        self.tree_name = tree
        self.recipe = recipe
        self.max_passes = max_passes
        self.passes = 0
        super().__init__(log_dir, log_file, tree=tree, recipe=recipe, **kwargs)
        self.tree.add_post_tick_handler(lambda t: self.environment.publish_state(self.tree_name, t, self.state))
        self.environment.reset_handler = self.reset_memory

    # ------------------------------------------------------------------ #
    #  Task setup                                                        #
    # ------------------------------------------------------------------ #
    def setup_state(self, recipe: Optional[str] = None, **kwargs):
        state = StretchState()
        if recipe:
            state.set_value("recipe/name", recipe)
            state.set_value("recipe/type", StretchState.load_recipe(recipe).get("type"))
        return state

    def setup_environment(self, **kwargs):
        return StretchEnvironment()

    def setup_tree(self, tree: str = DEFAULT_TREE, **kwargs):
        trees = available_trees()
        if tree not in trees:
            raise ValueError(f"Unknown tree '{tree}'. Trees in bt_cooking.stretch_bt: {sorted(trees)}")
        return py_trees.trees.BehaviourTree(root=trees[tree]())

    # Kept by reset_memory: the recipe is the task, and holding_object is what is
    # physically in the gripper, not a belief.
    KEPT_ON_RESET = ("recipe/name", "recipe/type", "holding_object")

    def reset_memory(self) -> str:
        '''
        Forgets what the robot has learned: every state variable except KEPT_ON_RESET goes
        back to its initial value (location beliefs back to the priors), and the current
        pass of the tree is abandoned so the next tick decides again from scratch.
        '''
        initial = StretchState().values
        for var, value in initial.items():
            if var not in self.KEPT_ON_RESET:
                self.state.set_value(var, value)
        self.tree.root.stop(Status.INVALID)
        return f"beliefs reset to the priors; kept {', '.join(self.KEPT_ON_RESET)}"

    # ------------------------------------------------------------------ #
    #  Injected Functions                                                #
    # ------------------------------------------------------------------ #

    def tree_done_condition(self, ros_node):
        '''
        Done once a pass of the tree finishes with nothing left to fetch ("target_object"
        is None), or after max_passes passes. A pass that fails with a target still set,
        e.g. the object was not where the robot looked, is retried on the next pass.
        '''
        if self.tree.root.status not in (Status.SUCCESS, Status.FAILURE):
            return False
        self.passes += 1
        target = self.state.get_value("target_object")
        ros_node.get_logger().info(
            f"Pass {self.passes} of '{self.tree_name}' finished with {self.tree.root.status}; target_object={target}")
        if target is None:
            return True
        if self.passes >= self.max_passes:
            ros_node.get_logger().warn(f"Stopping after {self.passes} passes with '{target}' still to fetch.")
            return True
        return False
