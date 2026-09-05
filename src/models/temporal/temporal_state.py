from dataclasses import dataclass
from typing import Dict


@dataclass
class AccountState:
    in_count: int = 0
    out_count: int = 0
    in_amount: float = 0.0
    out_amount: float = 0.0
    last_timestamp: int = -1

    @property
    def total_count(self) -> int:
        return self.in_count + self.out_count

    @property
    def avg_in_amount(self) -> float:
        if self.in_count == 0:
            return 0.0
        return self.in_amount / self.in_count

    @property
    def avg_out_amount(self) -> float:
        if self.out_count == 0:
            return 0.0
        return self.out_amount / self.out_count

    def time_since_last(self, timestamp: int) -> float:
        if self.last_timestamp < 0:
            return 0.0
        return max(0.0, float(timestamp - self.last_timestamp))

    def as_vector(self, timestamp: int):
        return [
            float(self.in_count),
            float(self.out_count),
            float(self.total_count),
            float(self.in_amount),
            float(self.out_amount),
            float(self.avg_in_amount),
            float(self.avg_out_amount),
            self.time_since_last(timestamp),
        ]

    def update_incoming(self, amount: float, timestamp: int) -> None:
        self.in_count += 1
        self.in_amount += float(amount)
        self.last_timestamp = timestamp

    def update_outgoing(self, amount: float, timestamp: int) -> None:
        self.out_count += 1
        self.out_amount += float(amount)
        self.last_timestamp = timestamp


class TemporalStateStore:
    def __init__(self):
        self._states: Dict[str, AccountState] = {}

    def get(self, account_id: str) -> AccountState:
        if account_id not in self._states:
            self._states[account_id] = AccountState()
        return self._states[account_id]

    def update_transaction(
        self,
        sender: str,
        receiver: str,
        amount_paid: float,
        amount_received: float,
        timestamp: int,
    ) -> None:
        self.get(sender).update_outgoing(
            amount=amount_paid,
            timestamp=timestamp,
        )

        self.get(receiver).update_incoming(
            amount=amount_received,
            timestamp=timestamp,
        )

    def __len__(self) -> int:
        return len(self._states)