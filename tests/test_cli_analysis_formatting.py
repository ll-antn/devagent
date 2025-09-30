from ai_dev_agent.cli.analysis.formatting import _get_main_argument
from ai_dev_agent.providers.llm.base import ToolCall


def _tool_call(name: str, diff: str) -> ToolCall:
    return ToolCall(name=name, arguments={"diff": diff})


def test_get_main_argument_write_patch_single_file():
    diff = """diff --git a/src/foo.py b/src/foo.py
--- a/src/foo.py
+++ b/src/foo.py
@@ -1,3 +1,4 @@
"""
    call = _tool_call("fs.write_patch", diff)
    assert _get_main_argument(call) == "src/foo.py"


def test_get_main_argument_write_patch_multiple_files():
    diff = """diff --git a/src/foo.py b/src/foo.py
--- a/src/foo.py
+++ b/src/foo.py
@@ -1,3 +1,4 @@
diff --git a/src/bar.py b/src/bar.py
--- a/src/bar.py
+++ b/src/bar.py
@@ -1,2 +1,3 @@
"""
    call = _tool_call("fs.write_patch", diff)
    assert _get_main_argument(call) == "2 files"


def test_get_main_argument_write_patch_deleted_file():
    diff = """diff --git a/src/old.py b/src/old.py
deleted file mode 100644
index e69de29..0000000
--- a/src/old.py
+++ /dev/null
@@ -1,3 +0,0 @@
"""
    call = _tool_call("fs.write_patch", diff)
    assert _get_main_argument(call) == "src/old.py"
