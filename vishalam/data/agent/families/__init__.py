"""Registry of all M1 task families (250: lists, strings, math, dicts, matrices, sorting,
recursion, parsing).

HELD_OUT families never appear in training data; they form the `test_unseen_family` split, which
measures whether the agent generalizes to kinds of tasks it has never practised.
"""
from __future__ import annotations

from .base import Case, Family, TaskInstance, make_instance  # noqa: F401
from .dict_tasks import FAMILIES as _DICTS
from .list_tasks import FAMILIES as _LISTS
from .math_tasks import FAMILIES as _MATH
from .misc_tasks import FAMILIES as _MISC
from .matrix_tasks import FAMILIES as _MATRICES
from .more_list_tasks import FAMILIES as _LISTS2
from .more_math_tasks import FAMILIES as _MATH2
from .more_string_tasks import FAMILIES as _STRINGS2
from .string_tasks import FAMILIES as _STRINGS

ALL: list[Family] = _LISTS + _STRINGS + _MATH + _DICTS + _MATRICES + _LISTS2 + _STRINGS2 + _MATH2 + _MISC
BY_ID: dict[str, Family] = {f.id: f for f in ALL}
assert len(BY_ID) == len(ALL), "duplicate family ids"

HELD_OUT = {
    # the original 11 (kept, so results stay comparable with the first 125M agent)
    "local_peaks", "alternating_sum", "every_nth", "longest_increasing_run",   # lists
    "interleave", "vowel_start_words", "mask", "sort_words",                   # strings
    "digital_root", "is_perfect", "round_to_multiple",                         # math
    # 28 more, spread over every category of the widened set
    "invert_dict", "anagram_groups", "letter_grades", "rename_keys",           # dicts
    "spiral_order", "rotate_90", "border_sum",                                  # matrices
    "majority_element", "merge_intervals", "leaders", "window_max",            # lists
    "to_roman", "is_valid_ipv4", "pig_latin", "wrap_words",                    # strings
    "catalan", "is_happy", "to_base", "prime_factors",                         # math
    "dense_rank", "custom_order", "median",                                    # sorting
    "power_set", "hanoi_moves", "coin_ways",                                   # recursion
    "is_valid_date", "compare_versions", "parse_duration",                     # parsing
}
TRAIN_FAMILIES = [f for f in ALL if f.id not in HELD_OUT]
UNSEEN_FAMILIES = [f for f in ALL if f.id in HELD_OUT]
