"""Qualified R0: rooted integer aperture-7 cell addresses (not Akitti physics).

Logical hierarchy only: physical hexagonal footprints are NOT necessarily nested.
No external dependencies, no file writes, no model weights, and no runtime globals mutated.
"""
from __future__ import annotations

import json
import math
from typing import Sequence

Matrix = tuple[tuple[int, int], tuple[int, int]]
Vec = tuple[int, int]
Address = tuple  # (depth, digit_at_level_1, ..., digit_at_level_depth)
A: Matrix = ((2, -1), (1, 3))
B: Matrix = ((3, 1), (-1, 2))
I: Matrix = ((1, 0), (0, 1))
# Seven symmetric hex-neighborhood representatives; complete cosets for BOTH A and B.
DIGITS: tuple[Vec, ...] = (
    (0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)
)
DEFAULT_MAX_DEPTH = 8
HARD_MAX_DEPTH = 12
MAX_COORD_BITS = 64


def _integer_pair(x: object) -> Vec:
    if not isinstance(x, (tuple, list)) or len(x) != 2:
        raise ValueError('coordinate must be a two-integer tuple/list')
    if any(type(v) is not int or abs(v).bit_length() > MAX_COORD_BITS for v in x):
        raise ValueError('coordinate must have two bounded integers')
    return x[0], x[1]


def matvec(m: Matrix, x: Vec) -> Vec:
    return (m[0][0] * x[0] + m[0][1] * x[1],
            m[1][0] * x[0] + m[1][1] * x[1])


def matmul(a: Matrix, b: Matrix) -> Matrix:
    columns = ((b[0][0], b[1][0]), (b[0][1], b[1][1]))
    v0, v1 = matvec(a, columns[0]), matvec(a, columns[1])
    return ((v0[0], v1[0]), (v0[1], v1[1]))


def determinant(m: Matrix) -> int:
    return m[0][0] * m[1][1] - m[0][1] * m[1][0]


def matrix_for_level(level: int) -> Matrix:
    if type(level) is not int or level < 0 or level >= HARD_MAX_DEPTH:
        raise ValueError('invalid refinement level')
    return A if level % 2 == 0 else B


def _exact_solve(m: Matrix, x: Vec) -> Vec:
    """Find q such that m*q=x; reject x not in m*Z^2."""
    companion = B if m == A else A if m == B else None
    if companion is None:
        raise ValueError('unknown lattice matrix')
    t = matvec(companion, x)
    if any(v % 7 for v in t):
        raise ValueError('coordinate not in transformed sublattice')
    return t[0] // 7, t[1] // 7


def _divide_one_level(x: Vec, m: Matrix) -> tuple[Vec, Vec]:
    matches: list[tuple[Vec, Vec]] = []
    for d in DIGITS:
        try:
            q = _exact_solve(m, (x[0] - d[0], x[1] - d[1]))
        except ValueError:
            continue
        matches.append((q, d))
    if len(matches) != 1:
        raise ValueError(f'expected exactly one residue digit, found {len(matches)}')
    return matches[0]


def check_core() -> None:
    """Raise on any broken algebraic invariant, including mutated matrices/digits."""
    if determinant(A) != 7 or determinant(B) != 7:
        raise ValueError('determinant must be seven')
    seven_i: Matrix = ((7, 0), (0, 7))
    if matmul(A, B) != seven_i or matmul(B, A) != seven_i:
        raise ValueError('A and B must be companion matrices')
    if len(DIGITS) != 7 or len(set(DIGITS)) != 7:
        raise ValueError('invalid digit set')
    for m in (A, B):
        classes = set()
        for d in DIGITS:
            # m*q has zero residue. The mod-7 image of adj(m)*d identifies the coset.
            other = B if m == A else A
            t = matvec(other, d)
            classes.add((t[0] % 7, t[1] % 7))
        if len(classes) != 7:
            raise ValueError('digits are not complete coset representatives')
    if matmul(A, B) != matmul(B, A):
        raise ValueError('failed orientation cycle')


class HexaflagLattice:
    """Rooted address tree. Valid addresses uniquely identify its descendants.

    At level n, child coordinate c_(n+1) = M_n * c_n + d, d in DIGITS.
    The seven digits are logical children, not perfect polygon subdivisions.
    """

    def __init__(self, max_depth: int = DEFAULT_MAX_DEPTH, root: Vec = (0, 0)):
        if type(max_depth) is not int or not 0 <= max_depth <= HARD_MAX_DEPTH:
            raise ValueError('invalid bounded max_depth')
        self.root = _integer_pair(root)
        self.max_depth = max_depth
        check_core()

    def validate(self, address: object) -> bool:
        if not isinstance(address, (tuple, list)) or len(address) < 1:
            return False
        depth = address[0]
        if type(depth) is not int or not 0 <= depth <= self.max_depth:
            return False
        if len(address) != depth + 1:
            return False
        for d in address[1:]:
            try:
                if _integer_pair(d) not in DIGITS:
                    return False
            except ValueError:
                return False
        return True

    def decode(self, address: Sequence) -> Vec:
        if not self.validate(address):
            raise ValueError('malformed address')
        current = self.root
        for level, d in enumerate(address[1:]):
            t = matvec(matrix_for_level(level), current)
            current = _integer_pair((t[0] + d[0], t[1] + d[1]))
        return current

    def decompose(self, coordinate: Vec, depth: int) -> tuple[Vec, Address]:
        """All Z^2 coordinates accepted within bounds; returns terminal quotient too.

        Retaining the quotient makes this decomposition information preserving.
        """
        if type(depth) is not int or not 0 <= depth <= self.max_depth:
            raise ValueError('invalid depth')
        x = _integer_pair(coordinate)
        reverse_digits: list[Vec] = []
        for level in range(depth - 1, -1, -1):
            x, digit = _divide_one_level(x, matrix_for_level(level))
            x = _integer_pair(x)
            reverse_digits.append(digit)
        return x, (depth, *reversed(reverse_digits))

    def compose(self, quotient: Vec, address: Sequence) -> Vec:
        """Inverse of decompose with any bounded quotient (not just self.root)."""
        if not self.validate(address):
            raise ValueError('malformed address')
        x = _integer_pair(quotient)
        for level, d in enumerate(address[1:]):
            y = matvec(matrix_for_level(level), x)
            x = _integer_pair((y[0] + d[0], y[1] + d[1]))
        return x

    def encode(self, coordinate: Vec, depth: int) -> Address:
        """Encode only descendants of this ROOT. Reject coordinates outside root tree.

        For arbitrary coordinates, use decompose() and retain the quotient.
        """
        quotient, address = self.decompose(coordinate, depth)
        if quotient != self.root:
            raise ValueError('coordinate is not a descendant of this root at this depth')
        return address

    def parent(self, address: Sequence) -> Address:
        if not self.validate(address) or address[0] == 0:
            raise ValueError('invalid address or root has no parent')
        return (address[0] - 1, *address[1:-1])

    def children(self, address: Sequence) -> tuple[Address, ...]:
        if not self.validate(address):
            raise ValueError('malformed address')
        if address[0] >= self.max_depth:
            return ()
        return tuple((address[0] + 1, *address[1:], d) for d in DIGITS)

    def transform(self, coordinate: Vec, level: int) -> Vec:
        if type(level) is not int or not 0 <= level < self.max_depth:
            raise ValueError('invalid level')
        return _integer_pair(matvec(matrix_for_level(level), _integer_pair(coordinate)))

    def dump_address(self, address: Sequence) -> str:
        if not self.validate(address):
            raise ValueError('malformed address')
        obj = {'version': 1, 'root': list(self.root), 'depth': address[0],
               'digits': [list(d) for d in address[1:]]}
        return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=True)

    def load_address(self, blob: str) -> Address:
        if type(blob) is not str or len(blob) > 4096:
            raise ValueError('invalid serialized address')
        obj = json.loads(blob)
        if type(obj) is not dict or set(obj) != {'version', 'root', 'depth', 'digits'}:
            raise ValueError('unsupported address fields')
        if type(obj['version']) is not int or obj['version'] != 1:
            raise ValueError('unsupported address version')
        if _integer_pair(obj['root']) != self.root:
            raise ValueError('address belongs to a different root')
        if type(obj['digits']) is not list:
            raise ValueError('invalid digits')
        address: Address = (obj['depth'], *[tuple(d) if isinstance(d, list) else d for d in obj['digits']])
        if not self.validate(address) or self.dump_address(address) != blob:
            raise ValueError('invalid or noncanonical address')
        return address

    def physical_center(self, address: Sequence) -> tuple[float, float]:
        """Optional planar visualization; not part of exact integer address identity.

        Coarsest triangular center spacing is one. At depth n, E_n=E0*(M0...Mn-1)^-1.
        """
        if not self.validate(address):
            raise ValueError('malformed address')
        c = self.decode(address)
        p = I
        for level in range(address[0]):
            p = matmul(p, matrix_for_level(level))
        det = determinant(p)
        u_num = p[1][1] * c[0] - p[0][1] * c[1]
        v_num = -p[1][0] * c[0] + p[0][0] * c[1]
        u, v = u_num / det, v_num / det
        return (u + 0.5 * v, math.sqrt(3) * 0.5 * v)


if __name__ == '__main__':
    check_core()
    lat = HexaflagLattice(max_depth=4)
    a = (2, DIGITS[1], DIGITS[4])
    xy = lat.decode(a)
    assert lat.encode(xy, 2) == a
    print('R0_REPAIR_SELFTEST=PASS')
