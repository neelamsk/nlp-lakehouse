from dataclasses import dataclass

@dataclass
class ClusterCostModel:
    node_hour_price: float
    dbu_rate_per_hour: float
    dbu_consumption_per_hour: float
    workers: int = 1
    include_driver: bool = True

def estimate_job_cost_usd(runtime_minutes: float, cost: ClusterCostModel) -> float:
    nodes = cost.workers + (1 if cost.include_driver else 0)
    hours = max(runtime_minutes, 0) / 60.0
    infra = nodes * cost.node_hour_price * hours
    dbu = cost.dbu_rate_per_hour * cost.dbu_consumption_per_hour * hours
    return infra + dbu

def cost_per_1k_predictions(total_predictions: int, runtime_minutes: float, cost: ClusterCostModel) -> float:
    total_cost = estimate_job_cost_usd(runtime_minutes, cost)
    if total_predictions <= 0:
        return 0.0
    return 1000.0 * (total_cost / total_predictions)
