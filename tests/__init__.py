# Marks tests/ as a package so `python -m unittest tests.test_x` and
# `unittest discover` both work. Without this, unittest reports a bare
# ModuleNotFoundError as "Ran 1 test ... FAILED (errors=1)", which reads like a
# genuine test failure but is only an import error.
