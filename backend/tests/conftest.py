"""Every test run gets an isolated DB/storage; never touch a user's corpus."""
import os
import tempfile

_data = tempfile.TemporaryDirectory(prefix="classscribe-tests-")
os.environ["CLASSSCRIBE_DATA_DIR"] = _data.name
os.environ["CLASSSCRIBE_DB_URL"] = f"sqlite:///{_data.name}/test.db"
