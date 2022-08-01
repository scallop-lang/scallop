from __future__ import annotations
from typing import Any, List, Optional, Tuple, Union


# Import internals
from .collection import ScallopCollection
from .context import ScallopContext
from .io import CSVFileOptions


class Relation:
    """
    The Relation class provides various overloaded operators for writing Scallop code
    in a pythonic way.

    For example, the following code:

    ```py
    ctx.add_relation("edge", (int, int))
    ctx.add_facts("edge", [(0, 1), (1, 2)])
    ctx.add_rule("path(a, c) = edge(a, c)")
    ctx.add_rule("path(a, c) = edge(a, b), path(b, c)")
    ```

    Can be written as:

    ```py
    edge = Relation(ctx, (int, int))
    path = Relation(ctx, (int, int))
    edge |= [(0, 1), (1, 2)]
    path["a", "c"] |= edge["a", "c"]
    path["a", "c"] |= edge["a", "b"] & path["b", "c"]
    ```
    """

    class __Rule:
        """Inner class for representing an atom of a scallop rule."""

        def __init__(self, ctx: ScallopContext, parts: List[str]) -> None:
            self.ctx = ctx
            self.parts = parts

        def __and__(self, other: "Relation.__Rule") -> "Relation.__Rule":
            assert self.ctx == other.ctx, "Relations must be in the same context"
            return self.__class__(self.ctx, self.parts + other.parts)

        def __ior__(lhs, rhs: "Relation.__Rule") -> None:
            """
            Overload the |= operator for adding a rule to the scallop context.

            Adds the rule: `lhs :- rhs` to the scallop context.

            :param lhs: The left hand side of the rule to add. (provided implicitly)
            :param rhs: The right hand side of the rule to add.
            """
            assert lhs.ctx == rhs.ctx, "Relations must be in the same context"
            assert len(lhs.parts) == 1, "LHS of |= must be a single relation"

            lhs.ctx.add_rule(f"{lhs} :- {rhs}")

        def __str__(self) -> str:
            return ", ".join(self.parts)

    ANON_COUNT = 0

    def __init__(
        self,
        ctx: ScallopContext,
        types: Union[Tuple, type, str],
        *,
        name: str = None,
        input_mapping: Optional[Union[List[Tuple], Tuple]] = None,
        retain_topk: Optional[int] = None,
        non_probabilistic: bool = False,
        load_csv: Optional[Union[CSVFileOptions, str]] = None,
    ):
        self.ctx = ctx
        if name is None:
            Relation.ANON_COUNT += 1
            name = f"relation_{Relation.ANON_COUNT}"
        self.name = name

        ctx.add_relation(
            relation_name=name,
            relation_types=types,
            input_mapping=input_mapping,
            retain_topk=retain_topk,
            non_probabilistic=non_probabilistic,
            load_csv=load_csv,
        )

    def __ior__(self, elems: List[Tuple]) -> Relation:
        """
        Overload the |= operator for adding facts for a relation.

        :param elems: The facts to add to the relation.
        """
        self.ctx.add_facts(self.name, elems)
        return self

    def __getitem__(self, key: Tuple) -> Relation.__Rule:
        stringify_key = ", ".join(str(k) for k in key)
        stringify_relation = f"{self.name}({stringify_key})"
        return Relation.__Rule(self.ctx, [stringify_relation])

    def __setitem__(self, key: Tuple, _: Any) -> None:
        """
        Overload the [] = operator to allow for the
        `r1["a"] |= r2["a"]`
        syntax.
        """
        pass

    def __iter__(self) -> ScallopCollection:
        return iter(self.ctx.relation(self.name))
