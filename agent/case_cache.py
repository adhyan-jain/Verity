"""
Bounded LRU cache for in-memory agent case/event state.
Person C: CASE_CACHE/EVENT_QUEUES in agent/api.py were plain unbounded dicts
keyed by caller-supplied case_id, so a stream of distinct case_ids grew
memory without limit for the life of the process. This caps that at
max_size entries, evicting the least-recently-used case first.
"""

from collections import OrderedDict
from collections.abc import Iterator
from typing import Any


class BoundedCaseCache:
    """Dict-like LRU cache with a hard cap on the number of entries."""

    def __init__(self, max_size: int = 500) -> None:
        self._max_size = max_size
        self._data: OrderedDict[str, Any] = OrderedDict()

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __getitem__(self, key: str) -> Any:
        value = self._data[key]
        self._data.move_to_end(key)
        return value

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value
        self._data.move_to_end(key)
        while len(self._data) > self._max_size:
            self._data.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def get(self, key: str, default: Any = None) -> Any:
        if key not in self._data:
            return default
        return self[key]
