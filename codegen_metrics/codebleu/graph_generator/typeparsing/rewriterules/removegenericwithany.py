from typing import Optional

from ..nodes import (
    ElipsisAnnotationNode,
    IndexAnnotationNode,
    SubscriptAnnotationNode,
    TupleAnnotationNode,
    TypeAnnotationNode,
    parse_type_annotation_node,
)
from .rewriterule import RewriteRule

__all__ = ["RemoveGenericWithAnys"]


class RemoveGenericWithAnys(RewriteRule):
    ANY_NODE = parse_type_annotation_node("typing.Any")

    def matches(self, node: TypeAnnotationNode, parent: Optional[TypeAnnotationNode]) -> bool:
        if not isinstance(node, SubscriptAnnotationNode):
            return False

        slice = node.slice
        if isinstance(slice, IndexAnnotationNode):
            slice = slice.value
        if isinstance(slice, ElipsisAnnotationNode):
            return True
        if isinstance(slice, TupleAnnotationNode):
            return all(s == self.ANY_NODE or isinstance(s, ElipsisAnnotationNode) for s in slice.elements)

        return False

    def apply(self, matching_node: TypeAnnotationNode) -> TypeAnnotationNode:
        return matching_node.value
