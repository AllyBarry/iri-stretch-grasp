from unaceba.task.state import State
from unaceba.utils.variables import CategoricalRange,ContinuousRange,BooleanRange,AnyRange,NormalisedFloatRange

class StretchState(State):

    _RECIPES = {
        "Salad":[
            "caprese_salad",
            "greek_salad",
            "chickpea_salad"
        ],
        "Sandwich":[
            "ham_cheese_tomato_sandwich",
            "cheese_mustard_sandwich"
        ]
    }

    def __init__(self, state_vars = [], var_ranges = {}, semantic_dict = {}, values = None):

        # Define state
        state_vars = [
            "recipe/type",
            "recipe/names",
        ]

        list(self._RECIPES.keys()),
        var_ranges = var_ranges | {
            "recipe/type":CategoricalRange(list(self._RECIPES.keys()),allow_none=True),
            "recipe/names":CategoricalRange([x for lst in self._RECIPES.values() for x in lst],allow_none=True)
        }

        semantic_dict = semantic_dict | {
            "recipe/type":"The broad category of recipe e.g. 'Salad' or 'Sandwich'",
            "recipe/names":"The specific recipe e.g. 'Greek Salad' or 'HamCheeseTomato Sandwich'",
        }

        # Populate with defaults
        if values is None:
            values = {key:None for key in state_vars}

        super().__init__(
            state_vars=state_vars,
            var_ranges=var_ranges,
            semantic_dict=semantic_dict,
            values=values
        )
