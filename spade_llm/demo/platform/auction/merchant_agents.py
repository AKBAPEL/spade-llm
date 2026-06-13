import asyncio
import logging
from typing import List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import Field

from spade_llm.core.agent import Agent
from spade_llm.core.behaviors import ContextBehaviour, MessageHandlingBehavior
from spade_llm.core.conf import Configurable, configuration
from spade_llm.demo.platform.auction.agent_prompts import (
    AGGRESSIVE_RESPONDER_PROMPT,
    TRUST_IMPRESSION_PROMPT,
)
from spade_llm.demo.platform.auction.bidder_behaviors import AuctionBidderBehaviour
from spade_llm.demo.platform.auction.dialogue_behaviors import DialogueResponderBehaviour
from spade_llm.demo.platform.auction.models import (
    AuctionProposal,
    BaseMerchantAgentConf,
)
from spade_llm.demo.platform.auction.scenario_loader import get_active_scenario
from spade_llm.demo.platform.auction.trust_mechanism import (
    CombinedTrustContextBuilder,
    PersonalTrustMechanism,
    SystemTrustClient,
)
from spade_llm.demo.platform.contractnet.discovery import AgentDescription, AgentTask, DF_ADDRESS

logger = logging.getLogger(__name__)


class FirstMerchantAgentConf(BaseMerchantAgentConf):
    bid_delay: float = Field(default=0, description="Delay between bids")


class SecondMerchantAgentConf(BaseMerchantAgentConf):
    bid_delay: float = Field(default=1, description="Delay between bids")


class ThirdMerchantAgentConf(BaseMerchantAgentConf):
    bid_delay: float = Field(default=1, description="Delay between bids")


class FourthMerchantAgentConf(BaseMerchantAgentConf):
    bid_delay: float = Field(default=1, description="Delay between bids")


class FifthMerchantAgentConf(BaseMerchantAgentConf):
    bid_delay: float = Field(default=1, description="Delay between bids")


class BaseMerchantAgent(Agent):
    """Base class for merchant agents with common auction logic."""

    shop_sku: dict = {}
    _agent_id: str = "base_merchant"
    _agent_description: str = "Агент-магазин который участвует в аукционе"
    _register_delay: float = 2.0
    _responder_prompt = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_bid: Optional[AuctionProposal] = None
        self.collaborators: Optional[List[str]] = None
        self.personal_trust: Optional[PersonalTrustMechanism] = None
        self.system_trust_client: Optional[SystemTrustClient] = None
        self._system_trust_model: Optional[BaseChatModel] = None

    def setup(self):
        """Initialize agent with auction and dialogue behaviors"""
        scenario = get_active_scenario()
        agent_cfg = scenario.agents.get(self._agent_id)
        if agent_cfg:
            self.shop_sku = agent_cfg.sku
            logger.info("Loaded SKU for %s from scenario: %d items", self._agent_id, len(self.shop_sku))
        else:
            logger.warning("No scenario config found for %s, using empty SKU", self._agent_id)
            self.shop_sku = {}

        # Backward compatibility with old config flag
        if getattr(self.config, "enable_trust_mechanism", False) and not self.config.enable_personal_trust:
            self.config.enable_personal_trust = True

        if self.config.enable_personal_trust:
            path = self.config.personal_trust_path or f"data/memory/{self._agent_id}_trust_memory.json"
            self.personal_trust = PersonalTrustMechanism(self._agent_id, path)
            logger.info("Personal trust enabled for %s", self._agent_id)

        asyncio.create_task(self.register_in_df())
        model = self.default_context.create_chat_model(self.config.model)
        self._system_trust_model = model
        self.add_behaviour(AuctionBidderBehaviour(
            config=self.config,
            model=model
        ))
        self.add_behaviour(DialogueResponderBehaviour(
            config=self.config,
            model=model,
            prompt=self._responder_prompt,
        ))

    async def get_partner_trust_context(self, partner_id: str, behavior: ContextBehaviour) -> str:
        """Build combined trust context for a partner."""
        system_score = None
        if self.config.enable_system_trust and self.system_trust_client is None:
            self.system_trust_client = SystemTrustClient(self.default_context, behavior)
        if self.system_trust_client is not None:
            system_score = await self.system_trust_client.get_partner_trust_score(partner_id)
        return CombinedTrustContextBuilder.build_context(
            partner_id=partner_id,
            personal=self.personal_trust,
            system_score=system_score,
        )

    def add_trust_impression(self, partner_id: str, impression: str, outcome: str, source: str = "runtime"):
        """Append a new trust impression record for a partner."""
        if self.personal_trust is not None:
            self.personal_trust.add_impression(partner_id, impression, outcome, source)
            self.personal_trust.save()
        else:
            logger.warning("Trust impression ignored: personal trust disabled for %s", self._agent_id)

    def get_latest_impression(self, partner_id: str) -> Optional[str]:
        """Return the latest impression text for a partner, or None."""
        if self.personal_trust is None:
            return None
        return self.personal_trust.get_latest_impression(partner_id)

    def save_trust_memory(self):
        """Save personal trust memory to disk."""
        if self.personal_trust is not None:
            self.personal_trust.save()

    def load_trust_memory(self):
        """Load personal trust memory from disk."""
        if self.personal_trust is not None:
            self.personal_trust.load()

    @property
    def trust_memory(self):
        """Backward-compatible accessor for personal trust memory."""
        if self.personal_trust is None:
            return {}
        return self.personal_trust.memory

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
    _agent_id = "first_merchant"
    _register_delay = 2.0


@configuration(SecondMerchantAgentConf)
class SecondMerchantAgent(BaseMerchantAgent, Configurable[SecondMerchantAgentConf]):
    """Agent representing the second merchant in the auction"""
    _agent_id = "second_merchant"
    _register_delay = 3.0


@configuration(ThirdMerchantAgentConf)
class ThirdMerchantAgent(BaseMerchantAgent, Configurable[ThirdMerchantAgentConf]):
    """Agent representing the 3 merchant in the auction"""
    _agent_id = "third_merchant"
    _register_delay = 3.0


@configuration(FourthMerchantAgentConf)
class FourthMerchantAgent(BaseMerchantAgent, Configurable[FourthMerchantAgentConf]):
    """Agent representing the 4th merchant — aggressive, unreliable partner."""
    _agent_id = "fourth_merchant"
    _register_delay = 3.0
    _responder_prompt = AGGRESSIVE_RESPONDER_PROMPT


@configuration(FifthMerchantAgentConf)
class FifthMerchantAgent(BaseMerchantAgent, Configurable[FifthMerchantAgentConf]):
    """Agent representing the 5th merchant in the auction"""
    _agent_id = "fifth_merchant"
    _register_delay = 3.0
