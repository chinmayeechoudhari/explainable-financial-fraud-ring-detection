"""Chronological interaction store for leakage-safe temporal neighborhoods."""
from __future__ import annotations

from bisect import bisect_left
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

    Events are appended in chronological order. Each per-node deque is kept in
    ascending timestamp order. Queries use binary search to find the prefix
    strictly earlier than the query timestamp, then scan only that prefix
    backwards to collect the most recent events.
    """

    def __init__(self, max_history_per_node: int = 64) -> None:
        if max_history_per_node <= 0:
            raise ValueError("max_history_per_node must be positive")
        self.max_history_per_node = max_history_per_node
        self._incoming: dict[int, deque[TemporalEvent]] = defaultdict(deque)
        self._outgoing: dict[int, deque[TemporalEvent]] = defaultdict(deque)
        self._incoming_timestamps: dict[int, deque[int]] = defaultdict(deque)
        self._outgoing_timestamps: dict[int, deque[int]] = defaultdict(deque)

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
            features=tuple(features),
        )
        outgoing = self._outgoing[event.source]
        outgoing_timestamps = self._outgoing_timestamps[event.source]
        incoming = self._incoming[event.destination]
        incoming_timestamps = self._incoming_timestamps[event.destination]

        outgoing.append(event)
        outgoing_timestamps.append(event.timestamp)
        incoming.append(event)
        incoming_timestamps.append(event.timestamp)

        while len(outgoing) > self.max_history_per_node:
            outgoing.popleft()
            outgoing_timestamps.popleft()
        while len(incoming) > self.max_history_per_node:
            incoming.popleft()
            incoming_timestamps.popleft()

    @staticmethod
    def _recent_before(
        history: deque[TemporalEvent],
        timestamps: deque[int],
        query_timestamp: int,
        limit: int,
    ) -> list[TemporalEvent]:
        """Return up to limit most-recent events with timestamp < query."""
        if not history:
            return []
        eligible_end = bisect_left(timestamps, int(query_timestamp))
        if eligible_end <= 0:
            return []
        start = max(0, eligible_end - int(limit))
        return list(history)[start:eligible_end][::-1]

    def get_history(
        self,
        node_id: int,
        query_timestamp: int,
        limit: int,
    ) -> tuple[list[TemporalEvent], list[TemporalEvent]]:
        """Return recent outgoing and incoming events strictly before T."""
        if limit <= 0:
            raise ValueError("limit must be positive")
        node_id = int(node_id)
        outgoing = self._recent_before(
            self._outgoing.get(node_id, deque()),
            self._outgoing_timestamps.get(node_id, deque()),
            int(query_timestamp),
            int(limit),
        )
        incoming = self._recent_before(
            self._incoming.get(node_id, deque()),
            self._incoming_timestamps.get(node_id, deque()),
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
