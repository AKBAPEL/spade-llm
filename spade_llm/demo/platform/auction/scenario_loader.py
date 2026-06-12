import logging
import os
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AgentScenarioConfig(BaseModel):
    sku: Dict[str, int] = Field(description="Agent's inventory with base prices")
    model: str = Field(default="max", description="Model name")
    bid_delay: float = Field(default=1.0, description="Delay between bids")
    enable_trust_mechanism: bool = Field(default=True, description="Enable trust memory")
    trust_preload_from_dialogues: bool = Field(default=False, description="Preload trust from dialogue files")
    register_delay: float = Field(default=3.0, description="Delay before DF registration")


class ScenarioConfig(BaseModel):
    name: str = Field(description="Scenario identifier")
    description: str = Field(description="Human-readable scenario description")
    user_request: List[str] = Field(description="List of ingredients the user wants")
    max_price: int = Field(description="User's maximum acceptable price")
    agents: Dict[str, AgentScenarioConfig] = Field(description="Agent configurations for this scenario")


# Global cache for the active scenario
_active_scenario: Optional[ScenarioConfig] = None


def _get_project_root() -> str:
    """Return the directory containing scenarios.yaml."""
    return os.path.dirname(os.path.abspath(__file__))


def _load_config_yaml() -> dict:
    """Load the main config.yaml to read the scenario key."""
    config_path = os.path.join(_get_project_root(), "config.yaml")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.warning("config.yaml not found at %s", config_path)
        return {}
    except Exception as e:
        logger.warning("Failed to load config.yaml: %s", e)
        return {}


def _load_scenarios_yaml() -> Dict[str, ScenarioConfig]:
    """Load all scenarios from scenarios.yaml."""
    scenarios_path = os.path.join(_get_project_root(), "scenarios.yaml")
    with open(scenarios_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    scenarios = {}
    for key, data in raw.get("scenarios", {}).items():
        scenarios[key] = ScenarioConfig.model_validate(data)
    return scenarios


def get_active_scenario() -> ScenarioConfig:
    """Return the currently active scenario.

    The scenario name is read from the 'scenario' key in config.yaml.
    If missing, defaults to 'scenario_a'.
    """
    global _active_scenario
    if _active_scenario is not None:
        return _active_scenario

    config = _load_config_yaml()
    scenario_name = config.get("scenario", "scenario_a")

    all_scenarios = _load_scenarios_yaml()
    if scenario_name not in all_scenarios:
        available = ", ".join(all_scenarios.keys())
        raise ValueError(
            f"Scenario '{scenario_name}' not found in scenarios.yaml. "
            f"Available: {available}"
        )

    _active_scenario = all_scenarios[scenario_name]
    logger.info(
        "Active scenario loaded: %s — %s",
        _active_scenario.name,
        _active_scenario.description,
    )
    return _active_scenario


def reload_scenario() -> ScenarioConfig:
    """Force reload the active scenario from disk."""
    global _active_scenario
    _active_scenario = None
    return get_active_scenario()


def list_scenarios() -> List[str]:
    """Return a list of available scenario names."""
    return list(_load_scenarios_yaml().keys())
