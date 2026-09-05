"""Chronological interaction store for leakage-safe temporal neighborhoods."""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class TemporalEvent:
    """One historical account interaction."""
    timestamp: int
    source: int
    destination: int
    features: tuple[float, ...]


class TemporalNeighborStore:
    """In-memory append-only event history with strict timestamp queries.

    Events are appended in chronological order. Queries use bisect-style
    backward traversal over per-node history and therefore return only events
    with event.timestamp < query_timestamp.
    """

    def __init__(self, max_history_per_node: int = 64) -> None:
        if max_history_per_node <= 0:
            raise ValueError("max_history_per_node must be positive")
        self.max_history_per_node = max_history_per_node
        self._incoming: dict[int, deque[TemporalEvent]] = defaultdict(deque)
        self._outgoing: dict[int, deque[TemporalEvent]] = defaultdict(deque)

    def add_event(
        self,
        timestamp: int,
        source: int,
        destination: int,
        features: Iterable[float],
    ) -> None:
        event = TemporalEvent(
            timestamp=int(timestamp),
            source=int(source),
            destination=int(destination),
            features=tuple(float(v) for v in features),
        )
        self._outgoing[event.source].append(event)
        self._incoming[event.destination].append(event)

        while len(self._outgoing[event.source]) > self.max_history_per_node:
            self._outgoing[event.source].popleft()
        while len(self._incoming[event.destination]) > self.max_history_per_node:
            self._incoming[event.destination].popleft()

    @staticmethod
    def _recent_before(
        history: deque[TemporalEvent],
        query_timestamp: int,
        limit: int,
    ) -> list[TemporalEvent]:
        """Return up to limit most-recent events with timestamp < query."""
        result: list[TemporalEvent] = []
        for event in reversed(history):
            if event.timestamp >= query_timestamp:
                continue
            result.append(event)
            if len(result) == limit:
                break
        return result

    def get_history(
        self,
        node_id: int,
        query_timestamp: int,
        limit: int,
    ) -> tuple[list[TemporalEvent], list[TemporalEvent]]:
        """Return recent outgoing and incoming events strictly before T."""
        if limit <= 0:
            raise ValueError("limit must be positive")
        outgoing = self._recent_before(
            self._outgoing.get(int(node_id), deque()),
            int(query_timestamp),
            int(limit),
        )
        incoming = self._recent_before(
            self._incoming.get(int(node_id), deque()),
            int(query_timestamp),
            int(limit),
        )
        return outgoing, incoming

    def recent_events(
        self,
        node_id: int,
        query_timestamp: int,
        limit: int,
    ) -> list[TemporalEvent]:
        """Return a merged recent temporal neighborhood strictly before T."""
        outgoing, incoming = self.get_history(node_id, query_timestamp, limit)
        events = outgoing + incoming
        events.sort(key=lambda item: item.timestamp, reverse=True)
        return events[:limit]

    def __len__(self) -> int:
        nodes = set(self._incoming) | set(self._outgoing)
        return len(nodes)
