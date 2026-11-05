# Exceptions

## try and except
Wrap code that may fail in a try block and handle errors in an except block naming the exception type, for example except ValueError. Avoid a bare except because it also catches KeyboardInterrupt and hides bugs.

## else and finally
The else block runs only if the try block raised no exception. The finally block always runs, whether or not an exception occurred, which makes it the place for cleanup such as closing resources.

## Raising exceptions
Use raise ValueError("message") to signal an error yourself. Inside an except block, raise NewError(...) from err keeps the original exception as the cause, which shows both tracebacks.

## Custom exceptions
Define your own exception by subclassing Exception, for example class ConfigError(Exception): pass. Custom exceptions let callers catch errors from your code specifically.
