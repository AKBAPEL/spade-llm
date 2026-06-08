import asyncio
import glob
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from spade_llm.core.agent import Agent
from spade_llm.core.behaviors import MessageHandlingBehavior
from spade_llm.core.conf import Configurable, configuration
from spade_llm.demo.platform.auction.agent_prompts import TRUST_IMPRESSION_PROMPT
from spade_llm.demo.platform.auction.bidder_behaviors import AuctionBidderBehaviour
from spade_llm.demo.platform.auction.dialogue_behaviors import DialogueResponderBehaviour
from spade_llm.demo.platform.auction.models import AuctionProposal, TrustImpressionRecord
from spade_llm.demo.platform.contractnet.discovery import AgentDescription, AgentTask, DF_ADDRESS

logger = logging.getLogger(__name__)


class FirstMerchantAgentConf(BaseModel):
    model: str = Field(description="Model name")
    bid_delay: float = Field(default=0, description="Delay between bids")
    enable_trust_mechanism: bool = Field(default=False, description="Enable trust memory and LLM-based impressions")
    trust_preload_from_dialogues: bool = Field(default=False, description="Preload trust impressions from existing dialogue files")


class SecondMerchantAgentConf(BaseModel):
    model: str = Field(description="Model name")
    bid_delay: float = Field(default=1, description="Delay between bids")
    enable_trust_mechanism: bool = Field(default=False, description="Enable trust memory and LLM-based impressions")
    trust_preload_from_dialogues: bool = Field(default=False, description="Preload trust impressions from existing dialogue files")


class ThirdMerchantAgentConf(BaseModel):
    model: str = Field(description="Model name")
    bid_delay: float = Field(default=1, description="Delay between bids")
    enable_trust_mechanism: bool = Field(default=False, description="Enable trust memory and LLM-based impressions")
    trust_preload_from_dialogues: bool = Field(default=False, description="Preload trust impressions from existing dialogue files")


class BaseMerchantAgent(Agent):
    """Base class for merchant agents with common auction logic."""

    shop_sku: dict = {}
    _agent_id: str = "base_merchant"
    _agent_description: str = "Агент-магазин который участвует в аукционе"
    _register_delay: float = 2.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_bid: Optional[AuctionProposal] = None
        self.collaborators: Optional[List[str]] = None
        self.trust_memory: Dict[str, List[TrustImpressionRecord]] = {}
        self._trust_memory_path = f"data/memory/{self._agent_id}_trust_memory.json"

    def setup(self):
        """Initialize agent with auction and dialogue behaviors"""
        asyncio.create_task(self.register_in_df())
        model = self.default_context.create_chat_model(self.config.model)
        self.add_behaviour(AuctionBidderBehaviour(
            config=self.config,
            model=model
        ))
        self.add_behaviour(DialogueResponderBehaviour(
            config=self.config,
            model=model
        ))
        self.load_trust_memory()

    def load_trust_memory(self):
        """Load trust memory from disk. Creates empty dict if file missing or corrupt.
        Migrates legacy flat string format to structured record list."""
        try:
            if os.path.exists(self._trust_memory_path):
                with open(self._trust_memory_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                migrated: Dict[str, List[TrustImpressionRecord]] = {}
                for partner, records in raw.items():
                    if isinstance(records, str):
                        # legacy single-string migration
                        migrated[partner] = [
                            TrustImpressionRecord(
                                timestamp=datetime.now().isoformat(),
                                impression=records,
                                outcome="unknown",
                                source="legacy"
                            )
                        ]
                    elif isinstance(records, list):
                        migrated[partner] = [
                            TrustImpressionRecord.model_validate(r) for r in records
                        ]
                    else:
                        migrated[partner] = []
                self.trust_memory = migrated
                total = sum(len(v) for v in self.trust_memory.values())
                logger.info("Loaded trust memory for %s: %d partners, %d total records", self._agent_id, len(self.trust_memory), total)
            else:
                self.trust_memory = {}
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load trust memory for %s: %s. Starting fresh.", self._agent_id, e)
            self.trust_memory = {}

    def save_trust_memory(self):
        """Save trust memory to disk atomically."""
        try:
            os.makedirs(os.path.dirname(self._trust_memory_path), exist_ok=True)
            tmp_path = self._trust_memory_path + ".tmp"
            serializable = {
                partner: [r.model_dump() for r in records]
                for partner, records in self.trust_memory.items()
            }
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(serializable, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self._trust_memory_path)
            total = sum(len(v) for v in self.trust_memory.values())
            logger.info("Saved trust memory for %s: %d partners, %d total records", self._agent_id, len(self.trust_memory), total)
        except OSError as e:
            logger.warning("Failed to save trust memory for %s: %s", self._agent_id, e)

    def add_trust_impression(self, partner_id: str, impression: str, outcome: str, source: str = "runtime"):
        """Append a new trust impression record for a partner."""
        record = TrustImpressionRecord(
            timestamp=datetime.now().isoformat(),
            impression=impression,
            outcome=outcome,
            source=source,
        )
        if partner_id not in self.trust_memory:
            self.trust_memory[partner_id] = []
        self.trust_memory[partner_id].append(record)

    def get_latest_impression(self, partner_id: str) -> Optional[str]:
        """Return the latest impression text for a partner, or None."""
        records = self.trust_memory.get(partner_id, [])
        if not records:
            return None
        return records[-1].impression

    async def register_in_df(self):
        """Register the agent in the directory facilitator"""
        context = self.default_context
        await asyncio.sleep(self._register_delay)
        await context.inform(DF_ADDRESS).with_content(self.create_description())

    def create_description(self) -> AgentDescription:
        """Create a description for the agent"""
        return AgentDescription(
            id=self._agent_id,
            description=self._agent_description,
            tasks=[
                AgentTask(
                    description="Получение текущего состояния аукциона и запрос на предложение",
                    examples=[
                        """
                        class ShopListRequest(BaseModel):
                            ingredients: List[str] = Field(description="List of ingredients needed")"""
                    ]
                )
            ]
        )


@configuration(FirstMerchantAgentConf)
class FirstMerchantAgent(BaseMerchantAgent, Configurable[FirstMerchantAgentConf]):
    """Agent representing the first merchant in the auction"""
    shop_sku = {
        "мясо (говядина)": 350,
        "свёкла": 50,
        "морковь": 35,
        "лук репчатый": 25,
        "капуста белокочанная": 85,
        "картофель": 45,
        "томатная паста": 40,
        "чеснок": 20,
        "уксус (лимонный сок)": 10,
        "лавровый лист": 5,
        "соль, перец": 10,
        "зелень (укроп/петрушка)": 30,
        # "сметана" : 60
    }
    _agent_id = "first_merchant"
    _register_delay = 2.0


@configuration(SecondMerchantAgentConf)
class SecondMerchantAgent(BaseMerchantAgent, Configurable[SecondMerchantAgentConf]):
    """Agent representing the second merchant in the auction"""
    shop_sku = {
        "мясо (говядина)": 320,
        "морковь": 30,
        "лук репчатый": 20,
        "капуста белокочанная": 80,
        "картофель": 40,
        "томатная паста": 35,
        "чеснок": 15,
        "лавровый лист": 5,
        "соль, перец": 10,
        "зелень (укроп/петрушка)": 25,
        "сметана": 60
    }
    _agent_id = "second_merchant"
    _register_delay = 3.0


@configuration(ThirdMerchantAgentConf)
class ThirdMerchantAgent(BaseMerchantAgent, Configurable[ThirdMerchantAgentConf]):
    """Agent representing the 3 merchant in the auction"""
    shop_sku = {
        # "мясо (говядина)": 550,
        "морковь": 30,
        "лук репчатый": 20,
        "капуста белокочанная": 80,
        "картофель": 40,
        "свёкла": 10,
        "уксус (лимонный сок)": 90,
        "томатная паста": 35,
        "чеснок": 15,
        "лавровый лист": 5,
        "соль, перец": 10,
        "зелень (укроп/петрушка)": 25,
        "сметана": 60
    }
    _agent_id = "third_merchant"
    _register_delay = 3.0
