from hydrodl2.src.api._version import __version__
from hydro_data_dev.api.methods import load_record, record, save_record
from hydro_data_dev.core.record import Record

# in case setuptools scm screw up and find version to be 0.0.0
assert not __version__.startswith("0.0.0")

__all__ = [
    "Record",
    "record",
    "save_record",
    "load_record",
]
