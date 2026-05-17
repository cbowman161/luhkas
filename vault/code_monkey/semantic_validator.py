from __future__ import annotations

import ast
import json
from typing import Any, Dict, List, Set

from .models import LocalModel
from .normalizer import extract_json_contract


def _module_functions_and_methods(source: str) -> List[str]:
    tree = ast.parse(source or "")
    names: List[str] = []
    class_stack: List[str] = []

    class Visitor(ast.NodeVisitor):
        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            class_stack.append(node.name)
            self.generic_visit(node)
            class_stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            if class_stack:
                names.append(class_stack[-1] + "." + node.name)
            else:
                names.append(node.name)
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            if class_stack:
                names.append(class_stack[-1] + "." + node.name)
            else:
                names.append(node.name)
            self.generic_visit(node)

    Visitor().visit(tree)
    return names


def _module_imports(source: str) -> List[str]:
    tree = ast.parse(source or "")
    names: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name:
                    names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module.split(".")[0])
            for alias in node.names:
                if alias.name and alias.name != "*":
                    names.add(alias.name)
    return sorted(names)


def _call_graph(source: str) -> Dict[str, List[str]]:
    tree = ast.parse(source or "")
    defined_bare: Set[str] = set()
    for name in _module_functions_and_methods(source):
        defined_bare.add(name.split(".")[-1])
    graph: Dict[str, Set[str]] = {}
    class_stack: List[str] = []
    function_stack: List[str] = []

    def current_name(node_name: str) -> str:
        if class_stack:
            return class_stack[-1] + "." + node_name
        return node_name

    class Visitor(ast.NodeVisitor):
        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            class_stack.append(node.name)
            self.generic_visit(node)
            class_stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            name = current_name(node.name)
            function_stack.append(name)
            graph.setdefault(name, set())
            self.generic_visit(node)
            function_stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            name = current_name(node.name)
            function_stack.append(name)
            graph.setdefault(name, set())
            self.generic_visit(node)
            function_stack.pop()

        def visit_Call(self, node: ast.Call) -> None:
            if function_stack:
                callee = None
                if isinstance(node.func, ast.Name):
                    callee = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    callee = node.func.attr
                if callee and callee in defined_bare:
                    graph.setdefault(function_stack[-1], set()).add(callee)
            self.generic_visit(node)

    Visitor().visit(tree)
    return {name: sorted(calls) for name, calls in sorted(graph.items())}


def _callers_from_graph(graph: Dict[str, List[str]]) -> Dict[str, List[str]]:
    callers: Dict[str, Set[str]] = {}
    for caller, callees in graph.items():
        for callee in callees:
            callers.setdefault(callee, set()).add(caller)
    return {name: sorted(values) for name, values in sorted(callers.items())}


def validate_readme_semantics(
    *,
    readme_content: str,
    main_content: str,
    tests_content: str = "",
    model: LocalModel | None = None,
) -> None:
    """Deterministically validate README/source semantic coverage.

    Step 23 removes the LLM-as-judge README gate. The previous semantic
    validator sometimes rejected adequate README files and then pushed the
    repair loop into progressively worse markdown. This validator keeps the
    same contract checks, but derives them from AST/source facts so a valid
    contract is repeatable. The unused ``model`` parameter is kept for API
    compatibility with Coder.
    """
    readme = readme_content or ""
    readme_low = readme.lower()
    source = main_content or ""
    issues: List[str] = []

    try:
        functions = _module_functions_and_methods(source)
        imports = _module_imports(source)
    except Exception as exc:
        raise ValueError("README semantic validation could not parse src/api.py: {}".format(exc)) from exc

    if "## purpose" not in readme_low or len(_section(readme, "## Purpose")) < 20:
        issues.append("README must explain the capability's purpose")
    if "## usage" not in readme_low or len(_section(readme, "## Usage")) < 20:
        issues.append("README must explain usage/calling style")

    for required in ["schema", "endpoint", "response"]:
        if required not in readme_low:
            issues.append("README must describe API-first contract including {}".format(required))

    function_section = _section(readme, "## Function Definitions").lower()
    public_section = _section(readme, "## Public API").lower()
    outputs_section = _section(readme, "## Outputs and Return Values").lower()
    imports_section = _section(readme, "## Imported Dependencies").lower()
    storage_section = _section(readme, "## Data Storage").lower()
    cleanup_section = _section(readme, "## Cleanup / Delete Behavior").lower()
    tests_section = _section(readme, "## Test Coverage").lower()

    for name in functions:
        bare = name.split(".")[-1].lower()
        if name.lower() not in function_section and bare not in function_section:
            issues.append("README Function Definitions missing {}".format(name))

    if functions:
        compact = function_section.replace(" ", "")
        if "calledby" not in compact:
            issues.append("README Function Definitions must include Called by information")
        if "calls:" not in function_section:
            issues.append("README Function Definitions must include Calls information")

    public_functions = _top_level_public_functions(source)
    for name in public_functions:
        low_name = name.lower()
        if low_name not in public_section and low_name not in function_section:
            issues.append("README must document public API function {}".format(name))
        if name != "schema" and low_name not in outputs_section:
            issues.append("README Outputs and Return Values must document {}".format(name))

    for name in imports:
        if name.lower() not in imports_section:
            issues.append("README Imported Dependencies missing {}".format(name))

    if "data_dir" in source.lower():
        for token in ["data_dir", "path(__file__).resolve().parent", "src/data"]:
            if token not in storage_section:
                issues.append("README Data Storage must explain {}".format(token))

    if any(word in source.lower() for word in ["delete", "remove", "cleanup", "unlink"]):
        if not any(word in cleanup_section for word in ["delete", "remove", "cleanup", "back-out"]):
            issues.append("README must explain cleanup/delete/back-out behavior through API endpoints")

    for token in ["schema", "endpoint", "response"]:
        if token not in tests_section:
            issues.append("README Test Coverage must mention {}".format(token))

    for key in ["ok", "action", "message", "data", "error"]:
        if key not in readme_low:
            issues.append("README must document response field {}".format(key))

    if issues:
        raise ValueError("README semantic validation failed: " + "; ".join(issues[:8]))


def _section(text: str, section: str) -> str:
    marker = section + "\n"
    idx = text.find(marker)
    if idx < 0:
        return ""
    start = idx + len(marker)
    import re
    match = re.search(r"^##\s+", text[start:], flags=re.MULTILINE)
    if match:
        return text[start:start + match.start()]
    return text[start:]


def _top_level_public_functions(source: str) -> List[str]:
    tree = ast.parse(source or "")
    top_level = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    names: List[str] = []
    if "schema" in top_level:
        names.append("schema")
    ignored = {"name", "package", "skill", "description", "purpose", "endpoints", "envelope", "response", "responses", "storage", "data", "version"}
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name != "schema":
            continue
        for child in ast.walk(node):
            if not isinstance(child, ast.Return) or child.value is None:
                continue
            try:
                value = ast.literal_eval(child.value)
            except Exception:
                continue
            if not isinstance(value, dict):
                continue
            endpoints = value.get("endpoints")
            if isinstance(endpoints, dict):
                schema_names = [str(k) for k in endpoints.keys() if isinstance(k, str)]
            elif isinstance(endpoints, (list, tuple, set)):
                schema_names = [str(item) for item in endpoints if isinstance(item, str)]
            else:
                schema_names = [str(k) for k, v in value.items() if isinstance(k, str) and k not in ignored and isinstance(v, dict)]
            names.extend(name for name in schema_names if name in top_level)
            break
    return sorted(set(names))
