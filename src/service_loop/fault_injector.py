"""Deterministic duplicate, reorder, and loss injection for Experiment 8A."""

from __future__ import annotations

import random


class FaultInjector:
    def __init__(self, seed: int = 20260810, loss_rate: float = 0.0, duplicate_rate: float = 0.0, reorder: bool = False) -> None:
        self.rng = random.Random(seed)
        self.loss_rate, self.duplicate_rate, self.reorder = loss_rate, duplicate_rate, reorder

    def apply(self, payloads: list[dict]) -> list[dict]:
        output = []
        for payload in payloads:
            if self.rng.random() >= self.loss_rate:
                output.append(payload)
                if self.rng.random() < self.duplicate_rate:
                    output.append(payload.copy())
        if self.reorder:
            self.rng.shuffle(output)
        return output
