"""
Definicje struktur danych używanych do opisu ról agentów.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class AgentRole:
    """
    Klasa opisująca rolę agenta w systemie multi‑agentowym.

    :param role_name: Nazwa roli (np. "Causal Analyst", "Creative Planner").
    :param expertise_areas: Lista dziedzin, w których agent się specjalizuje.
    :param thinking_style: Styl myślenia ("analytical", "creative", "critical", "systematic" itp.).
    """
    role_name: str
    expertise_areas: List[str]
    thinking_style: str