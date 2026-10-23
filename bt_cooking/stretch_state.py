import json
from pathlib import Path
from collections import defaultdict

from ament_index_python.packages import get_package_share_directory

from unaceba.task.state import State
from unaceba.utils.variables import CategoricalRange


def load_recipes():
    try:
        # Find the installed ROS 2 package.
        recipes_dir = (
            Path(get_package_share_directory("stretch_floor_perception"))
            / "config"
            / "recipes"
        )
    except Exception:
        # Fallback for execution directly from the source tree.
        recipes_dir = Path(__file__).resolve().parent.parent / "config" / "recipes"

    recipes = defaultdict(list)

    for recipe_file in sorted(recipes_dir.glob("*.json")):
        with recipe_file.open("r", encoding="utf-8") as f:
            recipe = json.load(f)

        recipe_id = recipe["recipe_id"]
        recipe_type = recipe["type"]

        recipes[recipe_type].append(recipe_id)

    return dict(recipes)


class StretchState(State):
    _RECIPES = load_recipes()

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

        if values is None:
            values = {key: None for key in state_vars}

        super().__init__(
            state_vars=state_vars,
            var_ranges=var_ranges,
            semantic_dict=semantic_dict,
            values=values,
        )