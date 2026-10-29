"""The interpreter's result as plain JSON: what other nodes consume. Pure Python (no ROS).

Only the task and its parameters, as on the whiteboard:
  {"task": "fetch", "params": {"object": "garlic", "from": null, "to": null}}
  {"task": "set_recipe", "params": {"recipe_name": "greek_salad", "recipe_type": "salad"}}
  {"task": "set_recipe", "params": {"recipe_name": null, "recipe_type": "salad"}}      # "make me a salad"

params lists every parameter of the task; null when the person did not say it. recipe_type is filled in
from recipe_name when a recipe is named. When there is nothing to do (unclear, not possible, not a
request) the result is {"task": null, "params": {}} and nothing is published on /speech/task_json.
Translation and confidence details stay in the interpreter's log.
"""


def task_json(result: dict, caps) -> dict:
    task = result.get('task') or {}
    name = task.get('name') if result.get('status') == 'execute' and task.get('name') else None
    if not name:
        return {'task': None, 'params': {}}
    action = caps.actions.get(name)
    params = {p: None for p in action.params} if action is not None else {}
    params.update({kv['key']: kv['value'] for kv in task.get('args', []) if kv.get('value') not in (None, '')})
    if name == 'set_recipe' and params.get('recipe_name'):
        params['recipe_type'] = recipe_type_of(params['recipe_name'], caps) or params.get('recipe_type')
    return {'task': name, 'params': params}


def recipe_type_of(recipe_name: str, caps):
    """The recipe's `type` from its file; without the files, the type named in the id (greek_salad -> salad)."""
    if recipe_name in getattr(caps, 'recipe_types', {}):
        return caps.recipe_types[recipe_name]
    action = caps.actions.get('set_recipe')
    types = (action.params['recipe_type'].enum or []) if action and 'recipe_type' in action.params else []
    words = str(recipe_name).lower().replace('-', '_').split('_')
    return next((t for t in types if t in words), None)
