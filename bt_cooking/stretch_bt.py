import py_trees

from bt_cooking.stretch_skills import *

def replan_sequence():
    reset_if_conditions_sequence = py_trees.composites.Sequence(
        name="reset_if_conditions_sequence",
        memory=False,
        children=[
            TargetNotOptimalCondition(),
            CanAbandonTaskCondition(),
            ResetTargetObjectNode()
        ]
    )
    
    reset_or_do_nothing_fallback = py_trees.composites.Selector(
        name="reset_or_do_nothing_fallback",
        memory=False,
        children=[
            reset_if_conditions_sequence,
            DoNothingNode()
        ]
    )
    
    decide_and_replan_sequence = py_trees.composites.Sequence(
        name="decide_and_replan_sequence",
        memory=False,
        children=[
            DecideTargetObjectNode(),
            reset_or_do_nothing_fallback
        ]
    )
    
    return decide_and_replan_sequence

def main_task_loop():
    navigate_if_not_location_fallback = py_trees.composites.Selector(
        name="navigate_if_not_location_fallback",
        memory=False,
        children=[
            AlreadyAtLocationCondition(),
            NavigateToPlaceNode(),
        ]
    )
    
    grasp_or_handle_failure_fallback = py_trees.composites.Selector(
        name="grasp_or_handle_failure_fallback",
        memory=True,
        children=[
            GraspObjectNode(),
            HandleGraspFailureNode()
        ]
    )
    
    grasp_nav_handover_sequence = py_trees.composites.Sequence(
        name="grasp_nav_handover_sequence",
        memory=True,
        children=[
            grasp_or_handle_failure_fallback,
            NavigateToHumanNode(),
            HandoverNode()
        ]
    )
    
    has_replanned_or_task_fallback = py_trees.composites.Selector(
        name="has_replanned_or_task_fallback",
        memory=True,
        children=[
            HasReplannedCheckAndSetNode(),
            grasp_nav_handover_sequence
        ]
    )
    
    main_task_loop_sequence = py_trees.composites.Sequence(
        name="decide_and_replan_sequence",
        memory=True,
        children=[
            DecideTargetLocationNode(),
            navigate_if_not_location_fallback,
            SearchForObjectNode(),
            replan_sequence(),
            has_replanned_or_task_fallback
        ]
    )
    
    return main_task_loop_sequence

def decision_making_loop():
    main_task_sequence = py_trees.composites.Sequence(
        name="main_task_sequence",
        memory=True,
        children=[
            CalculateReachabilityNode(),
            DecideTargetLocationNode(),
            main_task_loop(),
            UpdateRecipeStatusNode() # TODO: Implement
        ]
    )
    
    decision_making_selector = py_trees.composites.Selector(
        name="decision_making_selector",
        memory=False,
        children=[
            RecipeDoneCondition(),
            main_task_sequence
        ]
    )
    
    return decision_making_selector

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


def put_back_sequence():
    '''
    Puts back what the robot picked up by mistake (the grasp check found the wrong
    object): drives to where the last grasp picked it up and sets it down there. Each
    node does nothing unless a wrong object is held. main_demo_loop runs the same nodes
    at the start of every pass; this tree runs them on their own.
    '''
    return py_trees.composites.Sequence(
        name="put_back_sequence",
        memory=True,
        children=[
            NavigateToPickLocationNode(),
            MoveOverPickPointNode(),
            PlaceObjectNode(),
        ]
    )


def main_demo_loop():
    # Assumes "recipe/name" is already set. If the object is not found, the search
    # lowers that location's confidence, so the next pass searches somewhere else.
    demo_loop_seq = py_trees.composites.Sequence(
        name="demo_loop_seq",
        memory=True,
        children=[
            CalculateReachabilityNode(),
            DecideTargetObjectNode(),
            DecideTargetLocationNode(),
            NavigateToPlaceNode(),
            SearchForObjectNode(),
            GraspObjectNode(),
            NavigateToHumanNode(),
            HandoverNode(),
        ]
    )

    return demo_loop_seq
