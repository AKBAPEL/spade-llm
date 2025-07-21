import errno
import logging
import os
from collections import UserList
from pathlib import Path
from typing import Optional
from asyncio import sleep as asleep
import asyncio
import aiosqlite
from spade_llm.core.api import AgentContext
from aioconsole import ainput
from aiosqlite import Cursor, Connection
from async_lru import alru_cache
from langchain_chroma import Chroma
from langchain_community.document_loaders import CSVLoader
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel
from pydantic.fields import Field
from spade_llm.core.agent import Agent
from spade_llm.core.behaviors import MessageHandlingBehavior, MessageTemplate, ContextBehaviour
from spade_llm.core.api import Message
from spade_llm.demo.platform.contractnet.contractnet import ContractNetResponder, ContractNetRequest, \
    ContractNetProposal, \
    ContractNetResponderBehavior, ContractNetInitiatorBehavior
from spade_llm.demo.platform.contractnet.discovery import AgentDescription, AgentTask, DF_ADDRESS
from spade_llm.core.conf import configuration, Configurable
from spade_llm.demo import models
from spade_llm import consts
from typing import Union, Annotated, List, Tuple, Dict

logger = logging.getLogger(__name__)


class PlatformAgentConf(BaseModel):
    pass


class ShopList(BaseModel):
    """Plan to follow in future"""
    ingredients: Dict[str, int] = Field(
        description="ingredients dict with price",
    )


class ShopListRequest(BaseModel):
    """Request for shop list with ingredients"""
    ingredients: List[str] = Field(
        description="List of ingredients needed",
        default_factory=list
    )


class ShopListResponse(BaseModel):
    """Response with total price for ingredients"""
    total_price: int = Field(
        description="Total price for all ingredients",
        ge=0
    )


@configuration(PlatformAgentConf)
class PlatformAgent(Agent, Configurable[PlatformAgentConf]):
    class ShoplistRequestBehaviour(MessageHandlingBehavior):
        ingredients = [
            'мясо (говядина)',
            'свёкла',
            'морковь',
            'лук репчатый',
            'капуста белокочанная',
            'картофель',
            'томатная паста',
            'чеснок',
            'уксус (лимонный сок)',
            'лавровый лист',
            'соль, перец',
            'зелень (укроп/петрушка)',
            'сметана'
        ]

        def __init__(self, config: PlatformAgentConf):
            super().__init__(MessageTemplate.request())
            self.config = config

        async def step(self) -> None:
            # Create request model
            request = ShopListRequest(ingredients=self.ingredients)

            # Send request
            await (self.context.request("merchant")
                   .with_content(request))

            # Wait for response
            receiver = await self.receive(
                MessageTemplate(self.context.thread_id),
                timeout=25
            )

            # Parse response
            response = ShopListResponse.model_validate_json(receiver.content)

            # Reply with the received total price
            await self.context.reply_with_inform(self.message).with_content(
                f"Total price: {response.total_price} rub"
            )

    def setup(self):
        self.add_behaviour(self.ShoplistRequestBehaviour(self.config))


class MerchantAgentConf(BaseModel):
    pass


@configuration(MerchantAgentConf)
class MerchantAgent(Agent, Configurable[MerchantAgentConf]):


    class HandleRequestBehaviour(MessageHandlingBehavior):
        shop_sku = {
            "мясо (говядина)": 320,
            "свёкла": 50,
            "морковь": 30,
            "лук репчатый": 20,
            "капуста белокочанная": 80,
            "картофель": 40,
            "томатная паста": 35,
            "чеснок": 15,
            "уксус (лимонный сок)": 10,
            "лавровый лист": 5,
            "соль, перец": 10,
            "зелень (укроп/петрушка)": 25,
            "сметана": 60
        }
        def __init__(self, config: MerchantAgentConf):
            super().__init__(MessageTemplate.request())
            self.config = config

        async def step(self) -> None:
            # Parse incoming request
            request = ShopListRequest.model_validate_json(self.message.content)

            # Calculate total price
            total_price = sum(
                self.shop_sku.get(ingredient, 0)
                for ingredient in request.ingredients
            )

            # Create response model
            response = ShopListResponse(total_price=total_price)

            # Send response
            await (self.context
                   .reply_with_inform(self.message)
                   .with_content(response))

    def setup(self):
        self.add_behaviour(self.HandleRequestBehaviour(self.config))