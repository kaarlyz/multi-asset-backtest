from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, Optional


class BaseAgent(ABC):
    """Abstract base agent defining standard interface, evaluation, and logging signatures."""

    def __init__(self, agent_id: str, name: Optional[str] = None):
        self.agent_id = agent_id
        self.name = name or self.__class__.__name__
        self.logger = logging.getLogger(f"agent.{self.agent_id}")

    @abstractmethod
    def evaluate(self, context: Dict[str, Any]) -> Any:
        """
        Evaluate context at current_sim_time and produce typed analysis/decision artifact.
        Must strictly respect temporal invariants (no future leakage).
        """
        pass
