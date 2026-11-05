# Virtual environments and pip

## Creating a virtual environment
A virtual environment is an isolated folder with its own Python interpreter and installed packages. Create one with python -m venv .venv in the project directory.

## Activating
On Linux and macOS activate it with source .venv/bin/activate. On Windows run .venv\Scripts\activate. After activation, python and pip refer to the environment, and deactivate leaves it.

## Installing packages
Install a package with pip install requests. Pin exact versions for reproducibility with pip freeze > requirements.txt and install them later with pip install -r requirements.txt.

## Upgrading and removing
Upgrade a package with pip install --upgrade requests and remove one with pip uninstall requests. pip list shows all installed packages and their versions.
