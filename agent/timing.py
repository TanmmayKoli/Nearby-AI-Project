"""Per-node latency, collected with a LangChain callback (no changes to the nodes).

    timer = NodeTimer()
    run_turn(graph, thread_id, text, callbacks=[timer])
    timer.totals()  # {"extract": [1.2, 0.9], "ask_next": [2.1], ...}

A LangGraph node runs as a chain whose name equals metadata["langgraph_node"];
chains nested inside a node (e.g. the structured-output parser) have other
names, so they're ignored.
"""

import time
from collections import defaultdict
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler


class NodeTimer(BaseCallbackHandler):
    def __init__(self) -> None:
        self._started: dict[UUID, tuple[str, float]] = {}
        self.durations: dict[str, list[float]] = defaultdict(list)

    def on_chain_start(self, serialized: Any, inputs: Any, *, run_id: UUID, metadata: dict | None = None,
                       **kwargs: Any) -> None:
        node = (metadata or {}).get("langgraph_node")
        if node and node != "__start__" and kwargs.get("name") == node:  # __start__: LangGraph internal
            self._started[run_id] = (node, time.perf_counter())

    def _finish(self, run_id: UUID) -> None:
        if run_id in self._started:
            node, start = self._started.pop(run_id)
            self.durations[node].append(time.perf_counter() - start)

    def on_chain_end(self, outputs: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._finish(run_id)

    def on_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._finish(run_id)

    def totals(self) -> dict[str, list[float]]:
        return dict(self.durations)
