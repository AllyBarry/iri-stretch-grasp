import json
from pathlib import Path
from collections import defaultdict

from ament_index_python.packages import get_package_share_directory

from unaceba.task.state import State
from unaceba.utils.variables import CategoricalRange, NormalisedFloatRange

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
            "recipe/names",
        ]

        var_ranges = (var_ranges or {}) | {
            "recipe/type": CategoricalRange(
                list(self._RECIPES.keys()),
                allow_none=True
            ),
            "recipe/names": CategoricalRange(
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
            "recipe/names": (
                "The specific recipe ID, e.g. 'caprese_salad' or "
                "'ham_cheese_tomato_sandwich'"
            ),
        }

        # Location confidence variables: one per (object, location) pair
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

        if values is None:
            values = {
                key: confidence_defaults.get(key) for key in state_vars
            }

        super().__init__(
            state_vars=state_vars,
            var_ranges=var_ranges,
            semantic_dict=semantic_dict,
            values=values,
        )