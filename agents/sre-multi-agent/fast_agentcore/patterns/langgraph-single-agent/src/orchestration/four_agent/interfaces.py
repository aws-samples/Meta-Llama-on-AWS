"""Protocol definitions for four-agent orchestration components."""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from .schema import AgentMessage
from .state import IncidentState


@runtime_checkable
class ConversationAgent(Protocol):
    """Interface all agent implementations must satisfy."""

    name: str

    async def handle(
        self,
        incoming: AgentMessage,
        state: IncidentState,
    ) -> Optional[AgentMessage]:
        """Process a message and optionally emit a response."""


class AsyncCallableAgent(Protocol):
    """Convenience protocol for async call-style agent wrappers."""

    async def __call__(
        self,
        incoming: AgentMessage,
        state: IncidentState,
    ) -> Optional[AgentMessage]:
        ...


def ensure_agent(agent: ConversationAgent | AsyncCallableAgent) -> ConversationAgent:
    """Coerce callables into the ConversationAgent protocol."""

    if isinstance(agent, ConversationAgent):  # type: ignore[unreachable]
        return agent

    class _CallableWrapper:
        name = getattr(agent, "name", agent.__class__.__name__)

        async def handle(self, incoming: AgentMessage, state: IncidentState) -> Optional[AgentMessage]:
            return await agent(incoming, state)

    return _CallableWrapper()


__all__ = ["ConversationAgent", "AsyncCallableAgent", "ensure_agent"]
