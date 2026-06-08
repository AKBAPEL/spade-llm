import asyncio
import logging
from typing import List, Optional

from pydantic import BaseModel, Field

from spade_llm.core.agent import Agent
from spade_llm.core.behaviors import MessageHandlingBehavior
from spade_llm.core.conf import Configurable, configuration
from spade_llm.demo.platform.auction.bidder_behaviors import AuctionBidderBehaviour
from spade_llm.demo.platform.auction.dialogue_behaviors import DialogueResponderBehaviour
from spade_llm.demo.platform.auction.models import AuctionProposal
from spade_llm.demo.platform.contractnet.discovery import AgentDescription, AgentTask, DF_ADDRESS

logger = logging.getLogger(__name__)


class FirstMerchantAgentConf(BaseModel):
    model: str = Field(description="Model name")
    bid_delay: float = Field(default=0, description="Delay between bids")


class SecondMerchantAgentConf(BaseModel):
    model: str = Field(description="Model name")
    bid_delay: float = Field(default=1, description="Delay between bids")


class ThirdMerchantAgentConf(BaseModel):
    model: str = Field(description="Model name")
    bid_delay: float = Field(default=1, description="Delay between bids")


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

    def setup(self):
        """Initialize agent with auction and dialogue behaviors"""
        asyncio.create_task(self.register_in_df())
        self.add_behaviour(AuctionBidderBehaviour(
            config=self.config,
            model=self.default_context.create_chat_model(self.config.model)
        ))
        self.add_behaviour(DialogueResponderBehaviour(
            config=self.config,
            model=self.default_context.create_chat_model(self.config.model)
        ))

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
