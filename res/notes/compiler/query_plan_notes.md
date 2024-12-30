r(a, b) and s(a - b)
//
r(a, b) and s(d) and (d == a - b, b == a - d, a == b + d, d := a - b, b := a - d, a := b + d)
   1          2          3            4         ...



equivalent_classes: { {1}, {2}, {3, 4, 5, 6, 7, 8} }

arc:
- left: [l1, ..., ln], right: r

plan [a1, a2, ..., an]

constraints:
0. each atom only appears once in the plan
1. all the bounded variables in r should be bounded by l1, ..., ln
2. for all new variable v that r newly bounds
	2.1. if v is already bounded by l1, ..., ln, we rename the newly bounded variable into v', and we insert new constraint v == v'
	2.2. if v is not bounded already, then we add v to newly bounded variables

goal:
1. at least one atom in each equivalent class is selected





r(a, b) and s(d) and b := a - d
---1---     -2--     ----3-----
example: [1, 2, 3]
	[] -> [1]     =====> old: {}, new: {a, b}
	[1] -> [1, 2] =====> old: {a, b}, new: {d}
	[1, 2] -> [3] =====> old: {a, b, d}, new: {d'}, inserted constraint: {d == d'}

r(a, b) and s(d) and b == a - d
---1---     -2--     ----3-----
example: [1, 2, 3]
	[] -> [1]     =====> old: {}, new: {a, b}
	[1] -> [1, 2] =====> old: {a, b}, new: {a, b, d}
	[1, 2] -> [3] =====> old: {a, b, d},




fib(x - 1, a) and fib(x - 2, b) and (y == a + b, y := a + b, a == y - b, ...)
//
fib(U, a) and fib(V, b) and (U == x - 1, U := x - 1, x == U + 1, x := U + 1) and
														(V == x - 2, V := x - 2, x == V + 2, x := V + 2)

equivalent_classes: { {1}, {2}, {3, 4, 5, 6}, {7, 8, 9, 10} }

example:
  1. [] -> [1], old: {}, new: {U, a}
	2. [1] -> [1, ]




U == x - 1, U := x - 1, x == U + 1, x := U + 1
----1-----  ----2-----  ----3-----  ----4-----

2 <---> 4
^ <   > ^
|   x   |
1 /   \ 3
