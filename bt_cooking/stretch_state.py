from unaceba.task.state import State
from unaceba.utils.variables import CategoricalRange,ContinuousRange,BooleanRange,AnyRange,NormalisedFloatRange

class StretchState(State):

    def __init__(self, state_vars = [], var_ranges = {}, semantic_dict = {}, values = None):

        # Define state
        state_vars = []

        var_ranges = var_ranges | {

        }

        semantic_dict = semantic_dict | {

        }

        # Populate with defaults
        # TODO
        if values is None:
            values = {key:None for key in state_vars}

        super().__init__(
            state_vars=state_vars,
            var_ranges=var_ranges,
            semantic_dict=semantic_dict,
            values=values
        )
