"""Ограниченный калькулятор на Decimal. Никакого eval: только перечисленные операции."""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP, DivisionByZero, InvalidOperation

from .textnorm import to_decimal, fmt_decimal

OPS = {"percent_of", "multiply", "add", "subtract", "divide"}


class CalcError(ValueError):
    pass


def compute(operation: str, operands: list[str]) -> tuple[str, str]:
    """Возвращает (результат строкой, воспроизводимое выражение)."""
    if operation not in OPS:
        raise CalcError(f"неизвестная операция {operation!r}")
    nums = [to_decimal(x) for x in operands]
    if any(n is None for n in nums) or len(nums) < 2:
        raise CalcError("операнды не числа или их меньше двух")
    a, b = nums[0], nums[1]
    try:
        if operation == "percent_of":
            r, expr = a * b / Decimal(100), f"{fmt_decimal(a)} × {fmt_decimal(b)} % = "
        elif operation == "multiply":
            r, expr = a * b, f"{fmt_decimal(a)} × {fmt_decimal(b)} = "
        elif operation == "add":
            r, expr = sum(nums, Decimal(0)), " + ".join(fmt_decimal(n) for n in nums) + " = "
        elif operation == "subtract":
            r, expr = a - b, f"{fmt_decimal(a)} − {fmt_decimal(b)} = "
        else:
            r, expr = a / b, f"{fmt_decimal(a)} ÷ {fmt_decimal(b)} = "
    except (DivisionByZero, InvalidOperation) as e:
        raise CalcError(str(e)) from e
    r = r.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    res = fmt_decimal(r)
    return res, expr + res
