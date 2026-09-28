"""Bounded, conservative Python import analysis; never executes inspected code.

Tracks import aliases and simple assignment aliases/literal strings to a fixed
point. Bindings from different scopes/branches are conservatively unioned, so
ambiguous rebinding can reject safe code. This is not arbitrary dynamic-code
detection: computed code, reflection, and nonliteral module names are outside
the claim. Keep the two owned components' copies identical (parity tested) so
each remains usable without a new component dependency.
"""

import ast
from pathlib import Path
import tokenize


def imported_modules(source: str) -> set[str]:
    tree = ast.parse(source)
    nodes = list(ast.walk(tree))
    names = {"__import__": {"builtins.__import__"}}
    strings = {}
    modules = set()

    def resolve(node, bindings):
        if isinstance(node, ast.Name):
            return bindings.get(node.id, set())
        if isinstance(node, ast.Attribute):
            return {base + "." + node.attr for base in resolve(node.value, bindings)}
        return set()

    def literals(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return {node.value}
        if isinstance(node, ast.Name):
            return strings.get(node.id, set())
        return set()

    def bind(table, name, values):
        old = table.setdefault(name, set())
        size = len(old)
        old.update(values)
        return len(old) != size

    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
                bind(names, alias.asname or alias.name.split(".")[0],
                     {alias.name if alias.asname else alias.name.split(".")[0]})
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if base:
                modules.add(base)
            for alias in node.names:
                full = ".".join(filter(None, (base, alias.name)))
                modules.add(full)
                bind(names, alias.asname or alias.name, {full})

    # Bound the fixed point by assignment count. Only alias chains are resolved;
    # recursive expressions such as x = x.attr must not grow without limit.
    assignments = [n for n in nodes if isinstance(n, (ast.Assign, ast.AnnAssign))]
    for _ in range(len(assignments) + 1):
        changed = False
        for node in assignments:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            values = resolve(node.value, names)
            literal_values = literals(node.value)
            for target in targets:
                if isinstance(target, ast.Name):
                    changed |= bind(names, target.id, values)
                    changed |= bind(strings, target.id, literal_values)
        if not changed:
            break

    for node in nodes:
        if not isinstance(node, ast.Call):
            continue
        call_names = resolve(node.func, names)
        if not call_names.intersection({"builtins.__import__", "importlib.import_module"}):
            continue
        argument = node.args[0] if node.args else next(
            (kw.value for kw in node.keywords if kw.arg == "name"), None)
        modules.update(name.lstrip(".") for name in literals(argument))
    return modules


def check_import_boundary(package_dir, forbidden):
    """Return deterministic violations for resolvable imports and scan failures.

    Prohibitions denote Python module paths. Match at a dotted path boundary,
    including relative/package-qualified modules, never arbitrary substrings.
    Every Python file, including __init__, gates, and verifiers, is scanned.
    """
    root = Path(package_dir)
    violations = []
    try:
        files = sorted(root.rglob("*.py"))
        if not root.is_dir() or not files:
            return False, [f"{root}: no Python source to inspect"]
        for path in files:
            try:
                with tokenize.open(path) as source:
                    modules = imported_modules(source.read())
            except (OSError, UnicodeError, SyntaxError, ValueError) as exc:
                violations.append(f"{path}: source inspection failed ({type(exc).__name__})")
                continue
            for prohibition in sorted(forbidden):
                parts = prohibition.split(".")
                if any(
                    module.split(".")[start:start + len(parts)] == parts
                    for module in modules
                    for start in range(len(module.split(".")))
                ):
                    violations.append(f"{path}: imports {prohibition}")
    except OSError as exc:
        violations.append(f"{root}: source inspection failed ({type(exc).__name__})")
    return not violations, violations
