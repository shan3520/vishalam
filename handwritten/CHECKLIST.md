# Hand-written requests: your part of the dataset

You write **300+ requests** yourself, in your own words. This is the human-authored part of
Vishalam's data, and the test half is how we measure whether the agent understands the way real
people actually ask for things.

## Where and how

Put them in `handwritten/requests.jsonl`, one JSON object per line (copy the format from
`TEMPLATE.jsonl`):

```json
{"id": "h001", "split": "test", "category": "lists", "function_name": "second_smallest", "request": "Write a function `second_smallest(nums)` that returns the second smallest distinct number, or None if there isn't one."}
```

| field | meaning |
|---|---|
| `id` | `h001`, `h002`, ... never reuse an id |
| `split` | `test` (about 200, only used for scoring, never trained on) or `seed` (about 100, used as style examples for the rewording step) |
| `category` | `lists`, `strings` or `math` (M1 scope) |
| `function_name` | the exact function name, and it **must** appear in the request in backticks |
| `request` | your wording; say what to return in edge cases (empty input, ties, not found, ...) |

After you write them, I write the hidden tests for each `test` request (in `handwritten/hidden/`).
You check that the tests match what you meant.

## Rules that keep the score honest

1. **Your own words.** Don't paste from websites, LeetCode or AI tools, because that would defeat the
   purpose (and could leak benchmark problems).
2. **One function, plain inputs.** Arguments and return values are numbers, strings, lists, tuples,
   booleans or None. No files, no printing, no input().
3. **Name the function in backticks**: `` `my_function` `` or `` `my_function(a, b)` ``.
4. **Be precise about edge cases.** If the behaviour is ambiguous, the test is unfair.
5. **Vary your style.** Mix short and long requests, formal and casual, "write..." and "I need...", with
   and without the parameter list.
6. **Only a few near-copies of the generated families.** Covering similar ground is fine, but aim for
   at least half being tasks that are *not* in `vishalam/data/agent/families/`. That's where you
   learn whether the agent generalizes.

## Coverage checklist (aim for these rough counts across all 300)

- [ ] lists: 110 (filtering, searching, counting, reordering, pairs/windows, prefix/suffix, grouping into lists)
- [ ] strings: 110 (cleaning, counting, case changes, splitting/joining, simple parsing, validation)
- [ ] math: 80 (digits, divisibility, primes, sequences, conversions, simple geometry/rounding)
- [ ] at least 60 requests that do **not** give the parameter list
- [ ] at least 40 casual / non-native-style phrasings (that's how many real users write)
- [ ] at least 30 "fix the bug" requests: add `"kind": "fix"`, `"file": "utils.py"` (mention that file
      name in the request) and `"code": "def ...buggy code..."`. The code is placed in that file before
      the agent starts. Leave `kind` out for normal "write a function" requests.
