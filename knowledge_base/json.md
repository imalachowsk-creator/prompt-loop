# json

## Parsing JSON
json.loads() turns a JSON string into Python objects, and json.load() reads JSON directly from an open file object. JSON objects become dicts, arrays become lists, true and false become True and False, and null becomes None.

## Writing JSON
json.dumps() converts a Python object to a JSON string, and json.dump() writes it to an open file. Use indent=2 for human-readable output and ensure_ascii=False to keep non-ASCII characters such as Polish letters unescaped.

## Unsupported types
By default json cannot serialize objects such as datetime or set and raises TypeError. Pass default=str to json.dumps() to convert unknown objects to strings, or convert them yourself before serializing.

## Invalid input
When the input is not valid JSON, json.loads() raises json.JSONDecodeError, which is a subclass of ValueError. Catch it to report malformed data without crashing the program.
