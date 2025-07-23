import errno
import logging
import os
from collections import UserList
from pathlib import Path
from typing import Optional
from asyncio import sleep as asleep
import asyncio
import aiosqlite
from pydantic import ValidationError
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
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
    """List of ingredients with price"""
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
    missing_ingredients: List[str] = Field(
        description="List of ingredients that are not available",
        default_factory=list
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

            if response.missing_ingredients:
                # TODO: Make with failure
                await self.context.reply_with_inform(self.message).with_content(
                    f"Total price: {response.total_price} rub\n"
                    f"Missing ingredients: {', '.join(response.missing_ingredients)}"
                )
            else:
                await self.context.reply_with_inform(self.message).with_content(
                    f"Total price: {response.total_price} rub"
                )

    def setup(self):
        self.add_behaviour(self.ShoplistRequestBehaviour(self.config))


class MerchantAgentConf(BaseModel):
    model: str = Field(description="Model name")


class ConversationBehaviour(ContextBehaviour):
    def __init__(self, context: AgentContext, config: MerchantAgentConf, missing_ingredients: List[str]):
        super().__init__(context)
        self.config = config
        self.missing_ingredients = missing_ingredients

    async def step(self):
        request = ShopListRequest(ingredients=self.missing_ingredients)

        # Send request
        await (self.context.request("competitor")
               .with_content(request))

        self.set_is_done()


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
            #     "сметана": 60
            # }
        }

        def __init__(self, config: MerchantAgentConf,model: BaseChatModel):
            super().__init__(MessageTemplate.request())
            self.config = config
            self.model = model

        async def step(self) -> None:
            # Parse incoming request
            request = ShopListRequest.model_validate_json(self.message.content)

            # Find available and missing ingredients
            available_ingredients = []
            missing_ingredients = []

            for ingredient in request.ingredients:
                if ingredient in self.shop_sku:
                    available_ingredients.append(ingredient)
                else:
                    missing_ingredients.append(ingredient)

            # Calculate total price for available ingredients
            total_price = sum(
                self.shop_sku.get(ingredient, 0)
                for ingredient in available_ingredients
            )
            print("_____", missing_ingredients)
            final_missing = []

            if missing_ingredients:
                conversation = ConversationBehaviour(self.context, self.config, missing_ingredients)
                self.agent.add_behaviour(conversation)
                await conversation.join()
                competitor_response = await self.receive(template=MessageTemplate(thread_id=self.context.thread_id),
                                                         timeout=25)
                print(7777,competitor_response)
                try:
                    competitor_response = ShopList.model_validate_json(competitor_response.content)
                except ValidationError:
                    competitor_response = Conversate.model_validate_json(competitor_response.content)

                if isinstance(competitor_response, ShopList):
                    for ingredient in missing_ingredients:
                        if ingredient not in competitor_response.ingredients.keys():
                            final_missing.append(ingredient)
                        else:
                            total_price += competitor_response.ingredients[ingredient]
                    missing_ingredients.clear()
                elif isinstance(competitor_response, Conversate):
                    negotiation_prompt = ChatPromptTemplate.from_template(
                        """Ты - агент основного магазина. Ты получил контрпредложение от конкурента на недостающие ингредиенты. 
                        Твоя цель - получить необходимые товары на выгодных условиях, но ты можешь пойти на разумные уступки.

                        Исходный запрос клиента (недостающие ингредиенты):
                        {missing_ingredients}

                        Контрпредложение конкурента:
                        {competitor_offer}

                        Твои варианты действий:
                        1. Принять предложение конкурента (если оно разумное) - верни измененный запрос в формате ShopListRequest
                        2. Отклонить предложение (если оно невыгодное) - верни аргументированный отказ с объяснением почему конкурент должен пересмотреть свое предложение

                        При принятии решения учитывай:
                        - Важность этих ингредиентов для клиента
                        - Справедливость цены
                        - Возможность найти альтернативные варианты
                        - Репутационные риски

                        Respond in `json` format\n{format_instructions}. JSON only, without Markdown and additional text"""
                    )
                    self.parser = PydanticOutputParser(pydantic_object=Act)
                    s1 = negotiation_prompt | self.model
                    answer = await s1.ainvoke(
                        {
                            "missing_ingredients": missing_ingredients,
                            "competitor_offer": competitor_response.offer,
                            "format_instructions": self.parser.get_format_instructions(),
                        }
                    )

                    ans = self.parser.parse(answer.content)
                    print("+++", competitor_response.offer)
                    print("!!!", ans)
                    await (self.context.inform("competitor")
                           .with_content(ans.action))


            response = ShopListResponse(
                total_price=total_price,
                missing_ingredients=final_missing
            )

            # Send response
            await (self.context
                   .reply_with_inform(self.message)
                   .with_content(response))

    def setup(self):
        self.add_behaviour(self.HandleRequestBehaviour(config=self.config,
                                                       model=self.default_context.create_chat_model(self.config.model)))


class CompetitorAgentConf(BaseModel):
    model: str = Field(description="Model to use")


class Conversate(BaseModel):
    """Offer to another merchant"""
    offer: str = Field(
        description="List of ingredients needed"
    )


class Act(BaseModel):
    """Action to perform."""

    action: Union[ShopList, Conversate] = Field(
        description="Action to perform. "
                    "Если ты согласен предложить товары по списку из запроса то верни ShopList ингридиентами из запроса и ценами за которые ты готов их продать. "
                    "Если ты не согласен продать только эти товары то верни Conversate с развернутым и аргументированным предложением другому магазину в деловом стиле почему ты хочешь внести изменение в предложение запросившего."

    )


@configuration(CompetitorAgentConf)
class CompetitorAgent(Agent, Configurable[CompetitorAgentConf]):
    competitor_sku = {
        "лавровый лист": 8,
        "соль, перец": 12,
        "зелень (укроп/петрушка)": 30,
        "сметана": 70,
        "редька": 45,
        "петрушка корневая": 35
    }

    class HandleRequestBehaviour(MessageHandlingBehavior):
        def __init__(self, config: CompetitorAgentConf, model: BaseChatModel):
            super().__init__(MessageTemplate.request())
            self.config = config
            self.model = model
            self.parser = PydanticOutputParser(pydantic_object=Act)

        async def step(self) -> None:
            request = ShopListRequest.model_validate_json(self.message.content)
            # TODO: Ты соглашаешься отдать товары по запросу только если их купят вместе с другими товарами из твоего ассортимента.
            # TODO: Ты можешь увеличивать цены на товары которые нужны другим продавцам. 70rub->90rub
            merchant_prompt = ChatPromptTemplate.from_template(
                """Ты - агент магазина-конкурента. Ваша цель - максимизировать прибыль.
                 Ты соглашаешься отдать товары по запросу только если их купят вместе с другими товарами из твоего ассортимента
                Ваш ассортимент и цены:
                {competitor_sku}
                
                Получен запрос на следующие товары:
                {requested_items}

            Respond in `json` format\n{format_instructions}. JSON only, without Markdown and additional text"""
            )
            s = merchant_prompt | self.model
            answer = await s.ainvoke(
                {
                    "competitor_sku": self.agent.competitor_sku,
                    "requested_items": request.ingredients,
                    "format_instructions": self.parser.get_format_instructions(),
                }
            )

            ans = self.parser.parse(answer.content)
            print("!!!", ans)

            if isinstance(ans.action, ShopList):
                print(1)
                await self.context.reply_with_inform(self.message).with_content(ans.action)
            elif isinstance(ans.action, Conversate):
                print(2)
                await self.context.reply_with_inform(self.message).with_content(ans.action)
                competitor_response = await self.receive(template=MessageTemplate(thread_id=self.context.thread_id),
                                                         timeout=25)
                print(223,competitor_response)
                try:
                    competitor_response = ShopList.model_validate_json(competitor_response.content)
                except ValidationError:
                    competitor_response = Conversate.model_validate_json(competitor_response.content)

                print(1111,competitor_response)
                if isinstance(competitor_response, ShopList):
                    pass
                elif isinstance(competitor_response, Conversate):
                    merchant_prompt = ChatPromptTemplate.from_template(
                        """Ты - агент магазина-конкурента. Ваша цель - максимизировать прибыль.
                        
                        Ваш ассортимент и цены:
                        {competitor_sku}

                        Изначально был Получен запрос на следующие товары:
                        {requested_items}
                        
                        Но ты отказался от этого предложения и предложил следующее:
                        {ans}
                        
                        Агент конурента на это предложение ответил следующим образом:
                        {concur_offer}

                    Respond in `json` format\n{format_instructions}. JSON only, without Markdown and additional text"""
                    )
                    s2 = merchant_prompt | self.model
                    answer = await s2.ainvoke(
                        {
                            "ans": ans,
                            "concur_offer": competitor_response.offer,
                            "competitor_sku": self.agent.competitor_sku,
                            "requested_items": request.ingredients,
                            "format_instructions": self.parser.get_format_instructions(),
                        }
                    )

                    ans = self.parser.parse(answer.content)
                    print("!!!--", ans)
            else:
                print(3)

            self.set_is_done()

    def setup(self):
        self.add_behaviour(self.HandleRequestBehaviour(
            config=self.config,
            model=self.default_context.create_chat_model(self.config.model)
        ))
