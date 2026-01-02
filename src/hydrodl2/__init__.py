# src/hydrodl2/__init__.py
import logging
import os
import sys
from datetime import datetime
from importlib.resources import files
from pathlib import Path

from platformdirs import user_config_dir

from hydrodl2._version import __version__
from hydrodl2.api import available_models, available_modules, load_model, load_module

log = logging.getLogger('hydrodl2')

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
    package_name = 'hydrodl2'

    config_dir = Path(user_config_dir(package_name))
    agreement_file = config_dir / '.license_status'

    model_classes = available_models()

    if not agreement_file.exists():
        print(f"\n[----- {package_name} LICENSE AGREEMENT -----]")

        try:
            # Find and read LICENSE file
            license_path = files(package_name).parent.parent.joinpath("LICENSE")
            license = license_path.read_text(encoding="utf-8")
            print(license)
        except FileNotFoundError:
            # Fallback in case the LICENSE file wasn't packaged correctly
            print(
                "\n|> Error locating License. Showing summary <|\n"
                "By using this software, you agree to the terms specified \n"
                "in the Non-Commercial Software License Agreement: \n"
                "\nhttps://github.com/mhpi/hydrodl2/blob/master/LICENSE \n"
                "\n'hydrodl2' models are free for non-commercial use. \n"
                "Prior authorization must be obtained for commercial \n"
                "use. For further details, please contact the Pennsylvania \n"
                "State University Office of Technology Management at \n"
                "814.865.6277 or otminfo@psu.edu.\n"
            )

        print("\nThis agreement applies to all named models in this package:\n")

        if model_classes:  # Avoid error if model_classes is empty
            max_len = max(len(model) for model in model_classes) + 2
        else:
            max_len = 0

        for model in model_classes:
            model_string = ''
            for submodel in model_classes[model]:
                model_string += str(submodel) + ',  '
            print(f"-> {model:<{max_len}}: {model_string[:-3]}")

        print("-" * 40)

        response = input("Do you agree to these terms? Type 'Yes' to continue: ")

        if response.strip().lower() in ['yes', 'y']:
            try:
                config_dir.mkdir(parents=True, exist_ok=True)
                agreement_file.write_text(
                    f"accepted_on = {datetime.now().isoformat()}Z\nversion = 1\n",
                    encoding="utf-8",
                )
                log.warning(
                    f"License accepted. Agreement written to {agreement_file}\n"
                )
            except OSError as e:
                log.warning(f"Failed to save agreement file {agreement_file}: {e}")
                print(
                    "You may need to run with administrator privileges to avoid "
                    "repeating this process at runtime.",
                )
        else:
            print("\n>| License agreement not accepted. Exiting. <|")
            raise SystemExit(1)


def _should_skip_license():
    """Returns True if the license check should be bypassed."""
    # 1. Check if we are in a Non-Interactive shell (No user to type 'Yes')
    if not sys.stdin.isatty():
        return True

    # 2. Check for common 'Silent' or 'CI' flags
    if os.environ.get('CI') or os.environ.get('NGEN_SILENT'):
        return True

    # 3. Check for the Docker flag (as a fallback)
    if os.path.exists('/.dockerenv'):
        return True

    return False


# This only runs once when package is first imported.
if not _should_skip_license():
    _check_license_agreement()
