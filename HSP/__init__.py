"""
experiment scheduling module for autoHSP

- `hsp_{...}.py` scripts are not related to experiment scheduling, but rather used for solvent selection solely.
You should in principle be able to use them for general solvent selection tasks with the correct parameters.
- The other modules are coupled with the *autoHSP* lab experiment module and are used to schedule experiments.
    - `api.py`: the API wrappers for getting experiment-specific scheduling information and updating experiment records.
    - `tasks.py`: handling the scheduling of experiments, including sample prepration and image acquisition.
    - `utils.py`: utility functions for experiment scheduling.
"""

# require python >= 3.12
import sys

if sys.version_info < (3, 12):
    import warnings

    warnings.warn(
        (
            f"The `HSP` module requires Python 3.12 or higher. "
            f"You are using Python {'.'.join(map(str, sys.version_info[:3]))}. "
            f"There might be compatibility issues with older versions of Python."
        ),
        DeprecationWarning,
    )
