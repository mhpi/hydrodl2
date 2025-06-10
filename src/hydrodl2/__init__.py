import os
from importlib.resources import files
from pathlib import Path
from platformdirs import user_config_dir
from pathlib import Path

from hydrodl2._version import __version__
from hydrodl2.api import (available_models, available_modules,
                                  load_model, load_module)

# In case setuptools scm says version is 0.0.0
assert not __version__.startswith('0.0.0')

__all__ = [
    '__version__',
    'available_models',
    'available_modules',
    'load_model',
    'load_module',
]


def _check_license_agreement():
    """Checks if user has agreed to package license and prompts if not."""
    package_name = 'hydroDL2'

    config_dir = Path(user_config_dir(package_name))
    agreement_file = config_dir / '.license_accepted'

    if not agreement_file.exists():
        print(f"--- {package_name} LICENSE AGREEMENT ---")

        try:
            # Find and read LICENSE file
            license_text = files(package_name).joinpath("LICENSE").read_text(encoding="utf-8")
            print(license_text)
        except FileNotFoundError:
            # Fallback in case the LICENSE file wasn't packaged correctly
            print("\n[License file not found, showing summary.]")
            print("By using this software, you agree to the terms specified")
            print("in the Non-Commercial Software License Agreement.")
        
        print("-" * 40)

        # For this example, we'll just show a summary.
        print("\nBy using this software, you agree to the following terms:")
        print("1. You will use this for good, not evil.")
        print("2. You will tell your friends how cool this package is.")
        print("-" * 40)

        response = input("Do you agree to these terms? Type 'yes' to continue: ")

        if response.lower() == 'yes':
            try:
                config_dir.mkdir(parents=True, exist_ok=True)
                agreement_file.touch()
                print("\nThank you for agreeing. Enjoy the package!")
            except OSError as e:
                print(f"\nCould not save agreement file: {e}")
                print("You may need to run this once with administrator privileges.")
                raise SystemExit(1)
        else:
            print("\nLicense agreement not accepted. Exiting.")
            raise SystemExit(1) # Or raise a custom exception

# This code runs only once when the package is first imported
_check_license_agreement()
