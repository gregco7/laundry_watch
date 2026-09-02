"""Present so pytest puts the repo root on sys.path.

With the default (prepend) import mode, pytest inserts the first directory
above a test file that has no __init__.py -- which would be tests/, making
`import pipeline` fail. A conftest.py at the root adds the root instead.
Empty on purpose: fixtures live next to the tests that use them.
"""
