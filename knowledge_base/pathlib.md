# pathlib

## Creating paths
The pathlib module represents filesystem paths as Path objects instead of plain strings. Create one with Path("data/file.txt") and join parts with the slash operator, for example Path("data") / "file.txt". Path.home() returns the user's home directory and Path.cwd() returns the current working directory.

## Reading and writing files
Path.read_text() reads a whole file into a string and Path.write_text() writes a string to a file. Pass encoding="utf-8" to both to avoid platform-dependent defaults. For binary data use read_bytes() and write_bytes().

## Checking and listing
Path.exists() tells whether a path exists, is_file() and is_dir() check its type. Path.glob("*.csv") yields files matching a pattern in one directory, and rglob("*.csv") searches recursively in all subdirectories.

## Creating directories
Path.mkdir() creates a directory. Use mkdir(parents=True, exist_ok=True) to also create missing parent directories and to avoid an error if the directory already exists.

## Path parts
A Path exposes its components as attributes: name is the final component with extension, stem is the name without the extension, suffix is the extension such as ".txt", and parent is the containing directory.
