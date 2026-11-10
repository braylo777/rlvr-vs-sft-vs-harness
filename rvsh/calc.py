"""Safe arithmetic tool: evaluates assignment programs like
    apples = 48 / 2
    answer = apples + 48
using an AST whitelist. No names other than earlier assignments, no calls, no attributes."""
from __future__ import annotations

import ast
import math
import operator as op

_BIN = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
        ast.FloorDiv: op.floordiv, ast.Mod: op.mod, ast.Pow: op.pow}
_UN = {ast.USub: op.neg, ast.UAdd: op.pos}
_FUNCS = {"round": round, "abs": abs, "min": min, "max": max, "int": int, "float": float,
          "sqrt": math.sqrt, "ceil": math.ceil, "floor": math.floor}


class CalcError(ValueError):
    pass


def _eval(node: ast.AST, env: dict[str, float]) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        left, right = _eval(node.left, env), _eval(node.right, env)
        if isinstance(node.op, ast.Pow) and abs(right) > 12:
            raise CalcError("exponent too large")
        return _BIN[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UN:
        return _UN[type(node.op)](_eval(node.operand, env))
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        raise CalcError(f"unknown name {node.id}")
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS
            and not node.keywords and len(node.args) <= 3):
        return _FUNCS[node.func.id](*[_eval(a, env) for a in node.args])
    raise CalcError(f"disallowed syntax: {type(node).__name__}")


def run_program(src: str, max_lines: int = 40) -> float:
    """Run an assignment program; return the value of `answer` (or the last assignment)."""
    src = src.strip().removeprefix("```python").removeprefix("```").removesuffix("```")
    env: dict[str, float] = {}
    last = None
    lines = [ln.split("#")[0].rstrip() for ln in src.splitlines()]
    lines = [ln for ln in lines if ln.strip()][:max_lines]
    for ln in lines:
        try:
            tree = ast.parse(ln.strip())
        except SyntaxError:
            continue  # skip prose lines the model mixed in
        for stmt in tree.body:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                env[stmt.targets[0].id] = last = _eval(stmt.value, env)
            elif isinstance(stmt, ast.Expr):
                last = _eval(stmt.value, env)
            else:
                raise CalcError(f"disallowed statement: {type(stmt).__name__}")
    if "answer" in env:
        return env["answer"]
    if last is None:
        raise CalcError("no assignment found")
    return last


def format_number(x: float) -> str:
    if isinstance(x, float) and x.is_integer():
        x = int(x)
    if isinstance(x, float):
        return f"{x:.6f}".rstrip("0").rstrip(".")
    return str(x)
