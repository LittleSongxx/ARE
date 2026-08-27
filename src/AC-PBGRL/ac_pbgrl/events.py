from enum import IntFlag


class GraphEvent(IntFlag):
    NONE = 0
    FRONTIER_CHANGED = 1
    EDGE_INVALIDATED = 2
    REGION_SPLIT = 4
    REGION_MERGED = 8
    VISITED = 16
    NODE_REMOVED = 32

    @property
    def hard_reset(self) -> bool:
        return bool(
            self
            & (GraphEvent.VISITED | GraphEvent.NODE_REMOVED | GraphEvent.REGION_SPLIT | GraphEvent.REGION_MERGED)
        )
