# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Report unintentionally duplicate values in directly inherited enums."""

import ast
import io
import operator
import sys
import tokenize
from pathlib import Path

import regex as re

_UNKNOWN = object()
_ENUM_BASES = {"Enum", "IntEnum", "StrEnum"}
_FLAG_BASES = {"Flag", "IntFlag"}
_ALIAS_COMMENT = re.compile(r"# enum-alias: ok \((.*)\)\s*")
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Invert: operator.invert,
}
_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
    ast.BitOr: operator.or_,
    ast.BitAnd: operator.and_,
    ast.BitXor: operator.xor,
}


def _base_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "enum"
    ):
        return node.attr
    return None


def _enum_kind(node: ast.ClassDef) -> str | None:
    bases = {_base_name(base) for base in node.bases}
    if bases & _FLAG_BASES:
        return None
    for kind in ("StrEnum", "IntEnum", "Enum"):
        if kind in bases:
            return kind
    return None


def _is_member_name(name: str) -> bool:
    if name.startswith("__"):
        return False
    return not (
        len(name) > 2
        and name.startswith("_")
        and name.endswith("_")
        and not name.startswith("__")
        and not name.endswith("__")
    )


def _is_auto(node: ast.expr) -> bool:
    if not isinstance(node, ast.Call) or node.args or node.keywords:
        return False
    if isinstance(node.func, ast.Name):
        return node.func.id == "auto"
    return (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "auto"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "enum"
    )


def _has_top_level_auto(node: ast.expr) -> bool:
    return _is_auto(node) or (
        isinstance(node, ast.Tuple) and any(_is_auto(item) for item in node.elts)
    )


def _next_auto(kind: str, name: str, last_values: list[object]) -> object:
    if kind == "StrEnum":
        return name.lower()
    if any(value is _UNKNOWN for value in last_values):
        return _UNKNOWN
    if not last_values:
        return 1
    try:
        return operator.add(max(last_values), 1)
    except TypeError:
        pass
    for value in reversed(last_values):
        try:
            return operator.add(value, 1)
        except TypeError:
            pass
    return 1


def _safe_binary(op: ast.operator, left: object, right: object) -> object:
    if isinstance(op, ast.Pow) and isinstance(right, int) and abs(right) > 1024:
        return _UNKNOWN
    if (
        isinstance(op, (ast.LShift, ast.RShift))
        and isinstance(right, int)
        and not 0 <= right <= 4096
    ):
        return _UNKNOWN
    if (
        isinstance(op, ast.Mult)
        and isinstance(right, int)
        and abs(right) > 10000
        and isinstance(left, (str, bytes, list, tuple))
    ):
        return _UNKNOWN
    if (
        isinstance(op, ast.Mult)
        and isinstance(left, int)
        and abs(left) > 10000
        and isinstance(right, (str, bytes, list, tuple))
    ):
        return _UNKNOWN
    try:
        value = _BINARY_OPS[type(op)](left, right)
    except (ArithmeticError, TypeError, ValueError, MemoryError):
        return _UNKNOWN
    if isinstance(value, int) and value.bit_length() > 16384:
        return _UNKNOWN
    return value


def _evaluate(
    node: ast.expr,
    values: dict[str, object],
    kind: str,
    name: str,
    last_values: list[object],
    *,
    allow_auto: bool = False,
) -> object:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return values.get(node.id, _UNKNOWN)
    if _is_auto(node):
        if not allow_auto:
            return _UNKNOWN
        value = _next_auto(kind, name, last_values)
        last_values.append(value)
        return value
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        items = [
            _evaluate(
                item,
                values,
                kind,
                name,
                last_values,
                allow_auto=isinstance(node, ast.Tuple) and _is_auto(item),
            )
            for item in node.elts
        ]
        if any(item is _UNKNOWN for item in items):
            return _UNKNOWN
        try:
            if isinstance(node, ast.Tuple):
                return tuple(items)
            if isinstance(node, ast.List):
                return items
            return set(items)
        except TypeError:
            return _UNKNOWN
    if isinstance(node, ast.Dict):
        result = {}
        try:
            for key_node, value_node in zip(node.keys, node.values):
                if key_node is None:
                    return _UNKNOWN
                key = _evaluate(key_node, values, kind, name, last_values)
                value = _evaluate(value_node, values, kind, name, last_values)
                if key is _UNKNOWN or value is _UNKNOWN:
                    return _UNKNOWN
                result[key] = value
            return result
        except TypeError:
            return _UNKNOWN
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        operand = _evaluate(node.operand, values, kind, name, last_values)
        if operand is _UNKNOWN:
            return _UNKNOWN
        try:
            return _UNARY_OPS[type(node.op)](operand)
        except (ArithmeticError, TypeError, ValueError):
            return _UNKNOWN
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        left = _evaluate(node.left, values, kind, name, last_values)
        right = _evaluate(node.right, values, kind, name, last_values)
        if left is _UNKNOWN or right is _UNKNOWN:
            return _UNKNOWN
        return _safe_binary(node.op, left, right)
    return _UNKNOWN


def _comments(source: str) -> dict[int, str]:
    try:
        return {
            token.start[0]: token.string
            for token in tokenize.generate_tokens(io.StringIO(source).readline)
            if token.type == tokenize.COMMENT
        }
    except (tokenize.TokenError, SyntaxError):
        return {}


def _exempted(line: int, comments: dict[int, str]) -> bool:
    match = _ALIAS_COMMENT.fullmatch(comments.get(line, ""))
    return match is not None and bool(match.group(1).strip())


def _equal(left: object, right: object) -> bool:
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False


def _check_class(
    node: ast.ClassDef, kind: str, path: str, comments: dict[int, str]
) -> list[str]:
    errors = []
    members: list[tuple[str, object]] = []
    values: dict[str, object] = {}
    last_values: list[object] = []
    for statement in node.body:
        if isinstance(statement, ast.Assign):
            if not all(isinstance(target, ast.Name) for target in statement.targets):
                continue
            names = [
                target.id
                for target in statement.targets
                if isinstance(target, ast.Name)
            ]
            value_node = statement.value
        elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
            if not isinstance(statement.target, ast.Name):
                continue
            names = [statement.target.id]
            value_node = statement.value
        else:
            continue
        names = [name for name in names if _is_member_name(name)]
        if not names:
            continue

        automatic = _has_top_level_auto(value_node)
        working_last_values = last_values.copy()
        value = _evaluate(
            value_node,
            values,
            kind,
            names[0],
            working_last_values,
            allow_auto=_is_auto(value_node),
        )
        auto_values = working_last_values[len(last_values) :]
        for name in names:
            if value is not _UNKNOWN:
                for previous_name, previous_value in members:
                    if previous_value is not _UNKNOWN and _equal(value, previous_value):
                        if not _exempted(statement.lineno, comments):
                            errors.append(
                                f"{path}:{statement.lineno}: duplicate enum member "
                                f"{name!r} has the same value as {previous_name!r}"
                            )
                        break
            members.append((name, value))
            values[name] = value
            if automatic:
                last_values.extend(auto_values)
            else:
                last_values.append(value)
    return errors


def check_source(source: str, path: str = "<string>") -> list[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    comments = _comments(source)
    errors = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and (kind := _enum_kind(node)):
            errors.extend(_check_class(node, kind, path, comments))
    return errors


def check_file(path: str) -> list[str]:
    return check_source(Path(path).read_text(encoding="utf-8"), path)


def main() -> int:
    errors = [error for path in sys.argv[1:] for error in check_file(path)]
    for error in errors:
        print(error)
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
