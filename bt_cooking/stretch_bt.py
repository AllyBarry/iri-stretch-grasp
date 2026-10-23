import py_trees

from bt_cooking.stretch_skills import *

def test_sequence():
    test_seq = py_trees.composites.Sequence(
        name="test_sequence",
        memory=False,
        children=[
            TestNode(),
            CalculateReachabilityNode(),
            DecideTargetObjectNode()
        ]
    )

    return test_seq