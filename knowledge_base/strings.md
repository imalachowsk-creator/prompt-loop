# Strings and f-strings

## f-strings
An f-string is a string literal prefixed with f that evaluates expressions inside curly braces, for example f"Hello {name}". Since Python 3.8, f"{value=}" prints both the expression and its value, which helps debugging.

## Formatting numbers
Format specifiers go after a colon inside the braces. f"{x:.2f}" shows two decimal places, f"{n:,}" adds thousands separators, and f"{ratio:.1%}" shows a value as a percentage.

## Common string methods
str.split() breaks a string into a list, by default on whitespace. str.join() does the reverse, for example ", ".join(items). str.strip() removes leading and trailing whitespace, and str.replace(old, new) substitutes text.

## Immutability
Strings are immutable: methods like replace() and upper() return a new string and never change the original. Building a long string in a loop is faster with a list and a final join() than with repeated + concatenation.
