import ast
import logging
from typing import Any

logger = logging.getLogger(__name__)

_ALLOWED_BINOPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b, ast.Pow: lambda a, b: a ** b}
_MAX_POWER = 10


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _safe_eval(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(node.op, tuple(_ALLOWED_BINOPS.keys())):
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        if isinstance(node.op, ast.Pow):
            if abs(left) > _MAX_POWER or abs(right) > _MAX_POWER:
                raise ValueError("too large")
        if isinstance(node.op, ast.Div) and right == 0:
            raise ZeroDivisionError("division by zero")
        return _ALLOWED_BINOPS[type(node.op)](left, right)
    raise ValueError("unsafe expression")


def _contains_division(node: ast.AST) -> bool:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return True
    for child in ast.iter_child_nodes(node):
        if _contains_division(child):
            return True
    return False


def calculate_expression(expression: str) -> str:
    cleaned = expression.strip()
    if not cleaned:
        return "Please provide an arithmetic expression."

    try:
        logger.info("Calculator expression requested")
        parsed = ast.parse(cleaned, mode="eval")
        result = _safe_eval(parsed.body)
        if isinstance(result, float):
            if result.is_integer():
                if _contains_division(parsed.body):
                    return f"Result: {result:.1f}"
                return f"Result: {int(result)}"
            return f"Result: {result:.1f}" if abs(result) < 10 else f"Result: {result:.2f}"
        return f"Result: {result}"
    except ZeroDivisionError:
        return "Division by zero is not allowed."
    except ValueError as exc:
        if str(exc) == "too large":
            return "That expression is too large."
        return "Please use a simple arithmetic expression."
    except Exception:
        logger.exception("Calculator expression failed")
        return "Please use a simple arithmetic expression."
