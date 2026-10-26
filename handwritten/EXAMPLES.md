# Example hand-written requests

These show what good requests look like: different styles, categories and levels of detail. They
are **examples to learn the format from, not to copy**. Write your 300+ in your own words, about
your own ideas, in `handwritten/requests.jsonl` (rules and checklist: `CHECKLIST.md`).

Each line of `requests.jsonl` is one JSON object. The examples below are shown one per block so they
are easy to read; in the file, each one goes on a single line.

## What makes a request good

- **The function name is in backticks**, exactly as the agent must name it.
- **Edge cases are spelled out**: empty input, ties, "not found", negative numbers, case.
- **One clear answer per input.** If two people could reasonably return different things, add a
  sentence that settles it.
- **Plain inputs and outputs**: numbers, strings, lists, tuples, dicts, booleans, None.
- **Give 1–3 examples in most requests** (input → expected output), e.g. "`count_above_average([1, 2, 3, 10])`
  should return 1". The agent turns your examples into its tests; without them it can only check that
  its code runs. Leave them out in roughly 1 request in 6, so we also measure that case.

## Formal, with the parameter list

```json
{"id": "h001", "split": "test", "category": "lists", "function_name": "count_above_average",
 "request": "Write a function `count_above_average(scores)` that returns how many numbers in the list `scores` are strictly greater than the average of the list. Return 0 for an empty list."}
```

```json
{"id": "h002", "split": "test", "category": "strings", "function_name": "mask_email",
 "request": "Write `mask_email(address)`. Keep the first character of the part before the @, replace the rest of that part with asterisks, and keep everything from the @ on unchanged. For example 'maria@x.org' becomes 'm****@x.org'. If there is no @, return the string unchanged."}
```

## Casual, no parameter list

```json
{"id": "h003", "split": "test", "category": "math", "function_name": "minutes_to_clock",
 "request": "need a function `minutes_to_clock` - you give it a number of minutes since midnight and it gives back the time as 'HH:MM' with leading zeros. 75 -> '01:15'. assume the number is between 0 and 1439"}
```

```json
{"id": "h004", "split": "test", "category": "lists", "function_name": "last_index_of",
 "request": "can you write `last_index_of` that takes a list and a value and tells me the position of the LAST time that value shows up? if it isn't there give -1"}
```

## Non-native / informal phrasing (real users write like this a lot)

```json
{"id": "h005", "split": "test", "category": "strings", "function_name": "only_letters",
 "request": "please make function `only_letters(s)` which remove all thing that is not a letter from the string, keep the letters same case and same order"}
```

```json
{"id": "h006", "split": "test", "category": "math", "function_name": "sum_odd_digits",
 "request": "I want `sum_odd_digits(n)`, it should add only the odd digits of the number n. n can be negative also, then just ignore the minus sign"}
```

## Dicts, tuples, nested data

```json
{"id": "h007", "split": "test", "category": "dicts", "function_name": "cheapest_item",
 "request": "Write `cheapest_item(prices)` where `prices` is a dict mapping item names to prices. Return the name of the cheapest item. If two items share the lowest price, return the one whose name comes first alphabetically. Return None for an empty dict."}
```

```json
{"id": "h008", "split": "test", "category": "lists", "function_name": "pair_up",
 "request": "Write a function `pair_up(names)` that groups a list into tuples of two in order: ['a','b','c','d'] gives [('a','b'), ('c','d')]. If the list has an odd length, the last name goes in a tuple on its own, like ('e',)."}
```

## "Fix the bug" requests

Add `"kind": "fix"`, the file name in `"file"` (mention it in the request too) and the buggy code in
`"code"`. The code should look like something a person could really have written, with one
realistic mistake.

```json
{"id": "h009", "split": "test", "category": "lists", "kind": "fix", "file": "utils.py",
 "function_name": "average",
 "code": "def average(nums):\n    total = 0\n    for n in nums:\n        total += n\n    return total / len(nums) - 1\n",
 "request": "my `average` function in utils.py gives wrong answers, it should just return the mean of the list (and 0 for an empty list). can you fix it?"}
```

```json
{"id": "h010", "split": "test", "category": "strings", "kind": "fix", "file": "text_tools.py",
 "function_name": "count_words",
 "code": "def count_words(text):\n    return len(text.split(' '))\n",
 "request": "`count_words` in text_tools.py is supposed to count the words in a string, where words are separated by any amount of whitespace. Right now it gets it wrong for double spaces and for an empty string (which should be 0). Please fix it."}
```

## Seed requests (style examples, never used for scoring)

About 100 of your requests should have `"split": "seed"`. They are used only as examples of
real phrasing for the rewording step, so they can be simpler.

```json
{"id": "h011", "split": "seed", "category": "math", "function_name": "is_multiple_of_seven",
 "request": "write `is_multiple_of_seven(n)`: True if n divides by 7 with no remainder"}
```

```json
{"id": "h012", "split": "seed", "category": "strings", "function_name": "first_word",
 "request": "Give me a function called `first_word` that returns the first word of a sentence, or an empty string if there are no words."}
```

## Common mistakes to avoid

| Mistake | Why it is a problem | Better |
|---|---|---|
| "sort the list" | ascending or descending? in place or a new list? | "return a new list sorted from largest to smallest" |
| "find the duplicates" | each duplicate once? in what order? | "return each value that appears more than once, once, in order of first appearance" |
| "round the result" | to what? | "rounded to 2 decimal places" |
| "remove spaces" | all whitespace, or only ' '? leading/trailing only? | "remove every space character ' '" |
| "check if valid" | valid by which rules? | list the rules |

## Tips

- Write a few at a time, whenever you think of a task you have actually needed.
- Mix lengths: some one-liners, some with three or four sentences of detail.
- After you add `test` requests, I write their hidden tests in `handwritten/hidden/` and you check
  that each test matches what you meant.
