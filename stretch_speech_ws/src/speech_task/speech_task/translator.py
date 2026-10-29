"""Translate non-English requests to English before they are interpreted. Pure Python (no ROS).

Spoken requests: the transcriber detects the language (Transcript.language), so only non-English
utterances are translated. Typed requests carry no reliable language tag, so a cheap word check decides
whether the text already looks English; only text that does not is sent to the LLM.

The translation uses the same LLM backend as the interpreter, constrained to a small JSON schema, and is
told to use the robot's own English names for objects, places and recipes ("ajo" -> "garlic"), so the
interpreter, the grounding check and the stop phrases all keep working on English text.
"""
from __future__ import annotations

import re

from .backends import Backend, BackendError

# Typed text has no language tag, so it is classified by its function words: common English words against
# common Spanish/Catalan ones (the languages used here). Names ("pizza", "Doner Kebab") count for neither.
_ENGLISH = set("""
a an the this that these those some any every each no not none all both either neither other another such
i me my mine myself you your yours yourself he him his she her hers it its itself we us our ours they them their
am is are was were be been being do does did doing done have has had having will would shall should can could
may might must ought let lets let's please thanks thank hello hi hey goodbye bye yes yeah yep no nope okay ok sure
what which who whom whose where when why how whether if then than so because since while until unless though
and or but nor also too very really just only even still already again maybe perhaps quite rather almost
to of in on at by for from with without about into onto over under above below between through during after
before around near next up down out off away back here there now today tonight tomorrow soon later ever never
always usually often sometimes much many more most less least few little lot lots bit kind sort type way thing
something anything nothing everything someone anyone everyone somewhere anywhere one two three first last
get got give gave bring brought take took make made go went come came see saw look find found know knew think
thought want need like love prefer wonder tell told say said ask help try tried use put keep feel seem enjoy
good great nice best better bad fine fresh new old big small favorite favourite different same sure able
s t re ve ll d m don doesn didn isn aren wasn won wouldn couldn shouldn can't cannot
""".split())
_ROMANCE = set("""
el la lo los las un una unos unas de del al y o pero que qué con sin por para en es está están son soy estoy
me te se le les nos mi mis tu tus su sus mío este esta esto ese esa eso aquel muy más menos ya no sí también
dónde donde cuál cual cómo como cuando quién porque pues hay tengo tienes quiero quieres puedes puedo podrías
trae tráeme traeme dame pásame pasame busca encuentra dime hola gracias favor vale bueno ahora aquí allí
hacer haz hagamos vamos queremos ensalada bocadillo cocinar comer traer buscar quiero algo
els les uns unes amb però perquè què qui on quan com és són està estan sóc estic tinc vull vols pots puc
porta porta'm dona'm passa'm busca troba digues hola gràcies si us plau ara aquí allà fer fem anem amanida entrepà
""".split())


def looks_english(text: str, extra_words: set[str] = frozenset()) -> bool:
    """Cheap check for typed text: True if it is probably already English (not Spanish/Catalan)."""
    low = text.lower()
    if re.search(r"[áéíóúàèòïüñç¿¡·]", low):   # Spanish/Catalan letters and punctuation
        return False
    if re.search(r"(?:^|[\s(\"])[ldns]'[a-z]", low) or re.search(r"\b(?!i')[a-z]+(?<!n)'(?:m|t|ns|us|hi|ho|l|n)\b", low):
        return False   # Catalan elision / pronouns: l'all, d'oli, porta'm (not English I'm, don't, it'll)
    words = re.findall(r"[a-z]+", text.lower())   # "you're" -> you, re
    if not words:
        return True
    en = sum(w in _ENGLISH for w in words)
    ro = sum(w in _ROMANCE and w not in _ENGLISH for w in words)
    if en == 0 and ro == 0:   # only names / vocabulary words: no reason to translate
        return True
    return en > ro


TRANSLATE_SCHEMA = {
    'type': 'object',
    'properties': {
        'language': {'type': 'string', 'description': 'ISO 639-1 code of the input language, e.g. es, ca, en'},
        'english': {'type': 'string', 'description': 'the request in English'},
    },
    'required': ['language', 'english'],
    'additionalProperties': False,
}


class Translator:
    def __init__(self, backend: Backend, vocabulary: dict[str, list[str]], log=print):
        self.backend = backend
        self.log = log
        self.extra_words = {w for values in vocabulary.values() for v in values
                            for w in re.findall(r"[a-z']+", str(v).lower().replace('_', ' '))}
        listed = '\n'.join(f'- {name}: {", ".join(str(v).replace("_", " ") for v in values)}'
                           for name, values in vocabulary.items() if values)
        self.system_prompt = (
            'You translate short spoken requests to a home robot into English.\n'
            '- If the request is already in English, return it unchanged.\n'
            '- Translate the meaning naturally, as an English speaker would say it to the robot. Keep it short; '
            'do not add anything that was not said. Never answer, explain or reword it: questions stay '
            'questions, and you only translate.\n'
            '- When an object, place or recipe matches one of these English names, use exactly that name:\n'
            f'{listed}\n'
            '- Report the language of the input as an ISO 639-1 code.')

    def needs_translation(self, text: str, language: str = '', typed: bool = False) -> bool:
        if not text.strip():
            return False
        if typed or not language:
            return not looks_english(text, self.extra_words)
        return language.lower() not in ('en', 'english')

    def to_english(self, text: str) -> tuple[str, str]:
        """Returns (english_text, detected_language). Raises BackendError if the backend fails."""
        result = self.backend.interpret(self.system_prompt, [{'role': 'user', 'content': text}], TRANSLATE_SCHEMA)
        english = ' '.join(str(result.get('english', '')).split())
        if not english:
            raise BackendError(f'empty translation for {text!r}')
        return english, str(result.get('language', '')).lower()[:5]
