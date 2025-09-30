from ai_dev_agent.cli.analysis.formatting import _format_enhanced_tool_log, _get_main_argument
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


def test_get_main_argument_symbols_find_uses_name():
    call = ToolCall(name="symbols.find", arguments={"name": "numberLiteral", "lang": "cpp"})
    assert _get_main_argument(call) == "numberLiteral"


def test_format_enhanced_tool_log_includes_write_patch_target():
    diff = """diff --git a/checker/ETSAnalyzer.cpp b/checker/ETSAnalyzer.cpp
--- a/checker/ETSAnalyzer.cpp
+++ b/checker/ETSAnalyzer.cpp
@@ -1,3 +1,4 @@
"""
    call = _tool_call("fs.write_patch", diff)
    log = _format_enhanced_tool_log(
        call,
        repeat_count=1,
        execution_time=0.0,
        tool_output="Patch failed to apply",
        success=True,
    )
    assert log.startswith('⚡ fs.write_patch "checker/ETSAnalyzer.cpp" → Patch failed to apply')


def test_format_enhanced_tool_log_includes_symbols_find_name():
    call = ToolCall(name="symbols.find", arguments={"name": "numberLiteral", "lang": "cpp"})
    log = _format_enhanced_tool_log(
        call,
        repeat_count=1,
        execution_time=0.0,
        tool_output="No definitions found.",
        success=True,
    )
    assert log.startswith('🔣 symbols.find "numberLiteral" → No definitions found.')
