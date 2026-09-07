"""Symbolic building blocks for describing optimization problems.

Lets you write objectives and constraints the way you'd write them on
paper, e.g. ``2*x + y <= 10``, and get back an AST-like structure
(:class:`Expression`, :class:`Equation`) that solvers can consume.
"""

from dataclasses import dataclass
from enum import Enum, auto


class Op(Enum):
    LE = auto()
    LEQ = auto()
    GE = auto()
    GEQ = auto()
    EQ = auto()

    def __repr__(self):
        match self:
            case Op.LE:
                return "<"
            case Op.LEQ:
                return "<="
            case Op.GE:
                return ">"
            case Op.GEQ:
                return ">="
            case Op.EQ:
                return "="


@dataclass(frozen=True, slots=True)
class Expression:
    terms: dict

    def __repr__(self):
        return " + ".join(
            f"{coefficient}{var.symbol}"
            for var, coefficient in self.terms.items()
            if coefficient != 0
        ) or "0"

    @staticmethod
    def from_value(value):
        if isinstance(value, Expression):
            return value

        if isinstance(value, Var):
            return Expression({value: 1})

        if isinstance(value, (int, float)):
            return Expression({})

        raise TypeError(f"Cannot convert {type(value)} to Expression")

    def __add__(self, other):
        other = self.from_value(other)
        terms = self.terms.copy()

        for var, coefficient in other.terms.items():
            terms[var] = terms.get(var, 0) + coefficient

        return Expression(terms)

    def __sub__(self, other):
        other = self.from_value(other)
        terms = self.terms.copy()

        for var, coefficient in other.terms.items():
            terms[var] = terms.get(var, 0) - coefficient

        return Expression(terms)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Expression({
                var: coefficient * other
                for var, coefficient in self.terms.items()
            })

        return NotImplemented

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = self.from_value(other)
        if isinstance(other, (int, float)):
            return Expression({
                var: coefficient / other
                for var, coefficient in self.terms.items()
            })

        return NotImplemented

    def __floordiv__(self, other):
        other = self.from_value(other)
        if isinstance(other, (int, float)):
            return Expression({
                var: coefficient // other
                for var, coefficient in self.terms.items()
            })

        return NotImplemented

    def __le__(self, other):
        return Equation(self, Op.LEQ, other)

    def __ge__(self, other):
        return Equation(self, Op.GEQ, other)


@dataclass(frozen=True, slots=True)
class Equation:
    lhs: Expression
    op: Op
    rhs: float

    def __repr__(self):
        return f"{self.lhs} {self.op.__repr__()} {self.rhs}"


@dataclass(frozen=True, slots=True)
class Var:
    symbol: str

    def __mul__(self, other):
        return Expression({self: 1}) * other

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return Expression({self: other})
        return NotImplemented

    def __add__(self, other):
        return Expression({self: 1}) + other

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return Expression({self: 1}) - other

    def __rsub__(self, other):
        return -1 * self + other

    def __repr__(self):
        return self.symbol
