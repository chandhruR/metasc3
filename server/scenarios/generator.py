from __future__ import annotations

from typing import Optional

from server.state import CommunityState

from server.scenarios.task1_scenario import build_parenting_misinfo_community
from server.scenarios.task2_scenario import build_coordinated_harassment_community
from server.scenarios.task3_scenario import build_radicalization_pipeline_community


class CommunityGenerator:
    def generate_task1(self, seed: int = 42, n_accounts: Optional[int] = None) -> CommunityState:
        n = n_accounts if n_accounts is not None else 500
        return build_parenting_misinfo_community(seed, n_accounts=n)

    def generate_task2(self, seed: int = 42, n_accounts: Optional[int] = None) -> CommunityState:
        n = n_accounts if n_accounts is not None else 2000
        return build_coordinated_harassment_community(seed, n_accounts=n)

    def generate_task3(self, seed: int = 42, n_accounts: Optional[int] = None) -> CommunityState:
        n = n_accounts if n_accounts is not None else 5000
        return build_radicalization_pipeline_community(seed, n_accounts=n)

    def generate(
        self, task_id: str, seed: int = 42, n_accounts: Optional[int] = None
    ) -> CommunityState:
        if task_id == "task1_health_misinfo":
            return self.generate_task1(seed, n_accounts)
        if task_id == "task2_coordinated_harassment":
            return self.generate_task2(seed, n_accounts)
        if task_id == "task3_radicalization_pipeline":
            return self.generate_task3(seed, n_accounts)
        raise ValueError(f"Unknown task_id: {task_id}")
