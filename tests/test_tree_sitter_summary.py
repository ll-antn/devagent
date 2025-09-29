from pathlib import Path

import pytest

pytest.importorskip("tree_sitter_languages")

from ai_dev_agent.tools.code.code_edit.tree_sitter_analysis import TreeSitterProjectAnalyzer


def _parser_for_suffix(analyzer: TreeSitterProjectAnalyzer, suffix: str):
    parser = analyzer._get_parser_for_suffix(suffix)
    if parser is None:  # pragma: no cover - depends on optional dependency
        pytest.skip(f"tree-sitter parser for {suffix} not available")
    return parser


def test_typescript_summary_covers_common_declarations(tmp_path):
    analyzer = TreeSitterProjectAnalyzer(tmp_path)
    parser = _parser_for_suffix(analyzer, ".ts")

    source = """
    export class Foo extends Bar {}
    interface IFoo {}
    export const answer: number = 42;
    type Alias = string;
    enum Color { Red, Blue }
    """.strip()

    source_bytes = source.encode("utf-8")
    tree = parser.parse(source_bytes)

    outline = analyzer._summarize_file(Path("module.ts"), ".ts", tree, source_bytes)

    assert any("export class Foo" in entry for entry in outline)
    assert any("interface IFoo" in entry for entry in outline)
    assert any("export const answer" in entry for entry in outline)
    assert any("type Alias" in entry for entry in outline)
    assert any("enum Color" in entry for entry in outline)


def test_javascript_summary_includes_exports(tmp_path):
    analyzer = TreeSitterProjectAnalyzer(tmp_path)
    parser = _parser_for_suffix(analyzer, ".js")

    source = """
    export function greet(name) { return `hi ${name}`; }
    const helper = () => 42;
    export { helper };
    export default helper;
    """.strip()

    source_bytes = source.encode("utf-8")
    tree = parser.parse(source_bytes)

    outline = analyzer._summarize_file(Path("module.js"), ".js", tree, source_bytes)

    assert any("export function greet" in entry for entry in outline)
    assert any("const helper" in entry for entry in outline)
    assert any("export {" in entry for entry in outline)
    assert any("export default helper" in entry for entry in outline)


def test_c_summary_captures_functions_and_structs(tmp_path):
    analyzer = TreeSitterProjectAnalyzer(tmp_path)
    parser = _parser_for_suffix(analyzer, ".c")

    source = """
    int counter = 0;
    static inline int multiply(int a, int b) { return a * b; }

    struct Point {
        int x;
        int y;
    };

    typedef enum {
        RED = 1,
        BLUE,
    } Color;
    """.strip()

    source_bytes = source.encode("utf-8")
    tree = parser.parse(source_bytes)

    outline = analyzer._summarize_file(Path("module.c"), ".c", tree, source_bytes)

    assert any("function static inline int multiply" in entry for entry in outline)
    assert any("struct Point" in entry for entry in outline)
    assert any("member int x" in entry for entry in outline)
    assert any("enum Color" in entry for entry in outline)


def test_cpp_summary_handles_namespace_and_template(tmp_path):
    analyzer = TreeSitterProjectAnalyzer(tmp_path)
    parser = _parser_for_suffix(analyzer, ".cpp")

    source = """
    namespace demo {
        class Foo : public Bar {
            void method();
            int value;
        };

        template <typename T>
        T add(T a, T b) { return a + b; }

        using String = std::string;
    }
    """.strip()

    source_bytes = source.encode("utf-8")
    tree = parser.parse(source_bytes)

    outline = analyzer._summarize_file(Path("module.cpp"), ".cpp", tree, source_bytes)

    assert any("namespace demo" in entry for entry in outline)
    assert any("class Foo" in entry for entry in outline)
    assert any("member void method" in entry for entry in outline)
    assert any("template <typename T>" in entry for entry in outline)
    assert any("function T add" in entry for entry in outline)
    assert any("alias using String" in entry for entry in outline)
