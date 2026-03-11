from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
import random
from spade_llm.demo.platform.prisoners_dillema.agents import FIXED_AGENT_STRATEGY

@dataclass
class CommunityTopology:
    """
    Единая модель:
    • каждый агент принадлежит ровно одному сообществу
    • сообщества могут быть любого размера (включая 1)
    """
    allowed_edges: Set[Tuple[int, int]]
    community_of: Dict[int, int]                  # agent_id -> community_id
    community_strategy: Dict[int, Optional[str]]  # community_id -> strategy | None


# =========================
# 2. ГЕНЕРАТОР ГРАФА С СООБЩЕСТВАМИ (SBM)
# =========================

def generate_community_graph(
    communities: Dict[int, List[int]],
    p_intra: float = 0.8,
    p_inter: float = 0.2,
    community_strategy: Optional[Dict[int, Optional[str]]] = None,
    seed: Optional[int] = None
) -> CommunityTopology:
    """
    Stochastic Block Model:
    • p_intra — внутри сообщества
    • p_inter — между сообществами
    """
    rng = random.Random(seed)
    edges: Set[Tuple[int, int]] = set()

    community_of: Dict[int, int] = {
        agent: cid
        for cid, agents in communities.items()
        for agent in agents
    }

    agent_ids = list(community_of.keys())

    for i in agent_ids:
        for j in agent_ids:
            if i >= j:
                continue

            same_community = community_of[i] == community_of[j]
            p = p_intra if same_community else p_inter

            if rng.random() < p:
                edges.add((i, j))

    return CommunityTopology(
        allowed_edges=edges,
        community_of=community_of,
        community_strategy=community_strategy or {}
    )


# =========================
# 3. СЭМПЛИНГ ПАР
# =========================

def sample_pairs_from_topology(
    topology: CommunityTopology,
    max_pairs: Optional[int] = None
) -> List[Tuple[int, int]]:
    pairs = list(topology.allowed_edges)

    if max_pairs is None or max_pairs >= len(pairs):
        return pairs

    return random.sample(pairs, max_pairs)


# =========================
# 4. ФИКСИРОВАННАЯ СТРАТЕГИЯ ЧЕРЕЗ СООБЩЕСТВО
# =========================

def get_fixed_strategy(
    agent_id: int,
    topology: CommunityTopology
) -> Optional[str]:
    """
    Если у сообщества есть стратегия — агент её использует
    """
    if agent_id in FIXED_AGENT_STRATEGY:
        return FIXED_AGENT_STRATEGY[agent_id]

    cid = topology.community_of[agent_id]
    return topology.community_strategy.get(cid)