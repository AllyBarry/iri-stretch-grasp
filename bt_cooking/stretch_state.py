import json
from pathlib import Path
from collections import defaultdict
from typing import List,Dict,Any

from ament_index_python.packages import get_package_share_directory

from unaceba.task.state import State
from unaceba.utils.variables import CategoricalRange, NormalisedFloatRange, BooleanRange

_PKG = "stretch_floor_perception"
_CATEGORY_LABELS = {"Ingredients": "ingredient", "Tools": "tool"}


def load_recipe_data():
    recipes_dir = (
        Path(get_package_share_directory(_PKG)) / "config" / "recipes"
    )

    recipes = defaultdict(list)
    ingredients = set()
    tools = set()

    for recipe_file in sorted(recipes_dir.glob("*.json")):
        with recipe_file.open("r", encoding="utf-8") as f:
            recipe = json.load(f)

        recipe_id = recipe["recipe_id"]
        recipe_type = recipe["type"]

        recipes[recipe_type].append(recipe_id)
        ingredients.update(recipe.get("ingredients", []))
        tools.update(recipe.get("tools", []))

    return dict(recipes), sorted(ingredients), sorted(tools)


def load_location_data():
    priors_file = (
        Path(get_package_share_directory(_PKG))
        / "config"
        / "locations"
        / "location_priors.json"
    )

    with priors_file.open("r", encoding="utf-8") as f:
        priors = json.load(f)

    locations = []
    for objects in priors.values():
        for location_probs in objects.values():
            for location in location_probs:
                if location not in locations:
                    locations.append(location)

    return priors, locations


class StretchState(State):
    _RECIPES, _INGREDIENTS, _TOOLS = load_recipe_data()
    _LOCATION_PRIORS, _LOCATIONS = load_location_data()

    def __init__(self, state_vars=None, var_ranges=None,
                 semantic_dict=None, values=None):

        state_vars = [
            "recipe/type",
            "recipe/name",
        ]

        var_ranges = (var_ranges or {}) | {
            "recipe/type": CategoricalRange(
                list(self._RECIPES.keys()),
                allow_none=True
            ),
            "recipe/name": CategoricalRange(
                [
                    recipe_id
                    for recipe_ids in self._RECIPES.values()
                    for recipe_id in recipe_ids
                ],
                allow_none=True
            ),
        }

        semantic_dict = (semantic_dict or {}) | {
            "recipe/type": (
                "The broad category of recipe, e.g. 'salad' or 'sandwich'"
            ),
            "recipe/name": (
                "The specific recipe ID, e.g. 'caprese_salad' or "
                "'ham_cheese_tomato_sandwich'"
            ),
        }

        # Confidence in locations
        confidence_defaults = {}
        for category, objects in self._LOCATION_PRIORS.items():
            label = _CATEGORY_LABELS.get(category, category.lower())
            for obj, location_probs in objects.items():
                for location in self._LOCATIONS:
                    var = f"confidence/{obj}/{location}"
                    state_vars.append(var)
                    var_ranges[var] = NormalisedFloatRange()
                    semantic_dict[var] = (
                        f"The probability (0 to 1) that the {label} "
                        f"'{obj}' is located at the {location}"
                    )
                    confidence_defaults[var] = location_probs.get(location, 0.0)
        

        # Reachable and required
        availability_defaults = {}
        for obj in self._INGREDIENTS + self._TOOLS:
            vars = [
                f"reachable/{obj}",
                f"confidence/reachable/{obj}",
                f"required/{obj}",
                f"confidence/required/{obj}",
            ]
            state_vars += vars

            var_ranges[f"reachable/{obj}"] = BooleanRange()
            var_ranges[f"required/{obj}"] = BooleanRange()
            var_ranges[f"confidence/reachable/{obj}"] = NormalisedFloatRange()
            var_ranges[f"confidence/required/{obj}"] = NormalisedFloatRange()

            semantic_dict[f"reachable/{obj}"] = f"Whether or not object {obj} is reachable by the human"
            semantic_dict[f"required/{obj}"] = f"Whether or not object {obj} is required in the current recipe"
            semantic_dict[f"confidence/reachable/{obj}"] = f"The probability (0 to 1) that the variable 'reachable/{obj}' is correct"
            semantic_dict[f"confidence/required/{obj}"] = f"The probability (0 to 1) that the variable 'required/{obj}' is correct"

            availability_defaults[f"reachable/{obj}"] = False
            availability_defaults[f"required/{obj}"] = False
            availability_defaults[f"confidence/reachable/{obj}"] = 0.0
            availability_defaults[f"confidence/required/{obj}"] = 0.0

        

        # Values
        if values is None:
            values = {
                key: confidence_defaults.get(key) for key in state_vars
            }
            for var in availability_defaults:
                values[var] = availability_defaults[var]

        super().__init__(
            state_vars=state_vars,
            var_ranges=var_ranges,
            semantic_dict=semantic_dict,
            values=values,
        )

    '''
    UTILITY
    '''
    @staticmethod
    def load_recipe(recipe: str) -> Dict[str, Any]:
        """Load config/recipes/{recipe}.json from the package share directory."""
        path = Path(get_package_share_directory(_PKG)) / "config" / "recipes" / f"{recipe}.json"
        with open(path, "r") as f:
            return json.load(f)