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
                timeout=60
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
            self.set_is_done()

    def setup(self):
        self.add_behaviour(self.ShoplistRequestBehaviour(self.config))


"""

Communication agents

"""


class MerchantAgentConf(BaseModel):
    model: str = Field(description="Model name")


@configuration(MerchantAgentConf)
class MerchantAgent(Agent, Configurable[MerchantAgentConf]):
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

    class HandleRequestBehaviour(MessageHandlingBehavior):

        def __init__(self, config: MerchantAgentConf, model: BaseChatModel):
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
                if ingredient in self.agent.shop_sku:
                    available_ingredients.append(ingredient)
                else:
                    missing_ingredients.append(ingredient)

            # Calculate total price for available ingredients
            total_price = sum(
                self.agent.shop_sku.get(ingredient, 0)
                for ingredient in available_ingredients
            )

            final_missing = []

            if missing_ingredients:
                request = ShopListRequest(ingredients=missing_ingredients)
                thread = await self.context.fork_thread() # ВАЖНО! Для каждого агента свой тред
                # Send request
                await (thread.request("competitor")
                       .with_content(request))
                message_history = []
                # TODO: For i < self.max_iterations:
                # Probably should be fork thread for communication with each agent
                j = 0
                while True:
                    print(f"                 ITERATION IS {j}")
                    j+=1

                    competitor_response = await self.receive(template=MessageTemplate(thread_id=thread.thread_id),
                                                             timeout=60)
                    print("\nNOW RESPONSE IS\n",competitor_response.content)
                    try:
                        competitor_response = ShopList.model_validate_json(competitor_response.content)
                    except ValidationError:
                        competitor_response = Conversate.model_validate_json(competitor_response.content)

                    if isinstance(competitor_response, ShopList):
                        print("00000000000000000SHOP LIST IS INVOKED")
                        for ingredient in missing_ingredients:
                            if ingredient not in competitor_response.ingredients.keys():
                                final_missing.append(ingredient)
                            else:
                                total_price += competitor_response.ingredients[ingredient]
                        missing_ingredients.clear()
                        break
                    elif isinstance(competitor_response, Conversate):
                        negotiation_prompt = ChatPromptTemplate.from_template(
                            """Ты - агент основного магазина. Ты получил контрпредложение от конкурента на недостающие ингредиенты. 
                            Твоя цель - получить необходимые товары на выгодных условиях, но ты можешь пойти на различные уступки, например купить необходимые ингредиенты в рамках болельшего предложения.
                            
                            Ты можешь уступить конкуренту в надежде на сотрудничество в будущем.
                            Важно: За каждый виток переговоров твоя прибыль уменьшается на 20%. Чем быстрее договоришься - тем больше заработаешь.
                            Исходный запрос клиента (недостающие ингредиенты):
                            {missing_ingredients}
    
                            Контрпредложение конкурента:
                            {competitor_offer}
    
                            Тебе нужно взвесить все за и против и принять решение в данной ситуации:
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
                        request = ans.action
                        print("\n\n\n--------", request) #REQUEST IS INVOKING IN WHILE LOOP
                        await (thread.inform("competitor")
                               .with_content(request))

            response = ShopListResponse(
                total_price=total_price,
                missing_ingredients=final_missing
            )

            # Send response
            print("ReplY msg IS\n", self.message)
            await (self.context
                   .reply_with_inform(self.message)
                   .with_content(response))
            self.set_is_done()

    def setup(self):
        self.add_behaviour(self.HandleRequestBehaviour(config=self.config,
                                                       model=self.default_context.create_chat_model(self.config.model)))


class CompetitorAgentConf(BaseModel):
    model: str = Field(description="Model to use")


class Conversate(BaseModel):
    """Offer to another merchant"""
    offer: str = Field(
        description="деловое предложение конкуренту в формате строки без Markdown"
    )


class Act(BaseModel):
    """Action to perform."""

    action: Union[ShopList, Conversate] = Field(
        description="Action to perform. "
                    "Если ты согласен предложить товары по списку из запроса то верни ShopList с ингредиентами и ценами за которые ты готов их продать. "
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
            self.conversation_history = []  # Хранит историю взаимодействий

            # Единый универсальный промпт
            self.merchant_prompt = ChatPromptTemplate.from_template(
                """Ты - агент магазина-конкурента. Ваша цель - максимизировать прибыль, учитывая что:
                    - За каждый раунд переговоров ваша потенциальная прибыль уменьшается на 20%
                    - Быстрые сделки приносят больше чистой прибыли
                Ты соглашаешься отдать товары по запросу только если их купят вместе с другими товарами из твоего ассортимента, но ты можешь пойти на различные уступки.
                Текущий контекст переговоров:
                {conversation_history}

                Ваш ассортимент и цены:
                {competitor_sku}

                Текущий запрос/предложение:
                {current_interaction}
                
                Тебе нужно взвесить все за и против и принять решение в данной ситуации:
                    1. Принять предложение конкурента (если оно разумное) - верни измененный запрос в формате ShopList.
                    2. Отклонить предложение (если оно невыгодное) - верни аргументированный отказ с объяснением почему конкурент должен пересмотреть свое предложение
                
                Respond in `json` format\n{format_instructions}. JSON only, without Markdown and additional text"""
            )

        def _update_history(self, role: str, content: str):
            """Обновляет историю переговоров"""
            self.conversation_history.append(f"{role}: {content}")
            if len(self.conversation_history) > 5:  # Ограничиваем размер истории
                self.conversation_history.pop(0)

        async def process_interaction(self, interaction_data: dict) -> Act:
            """Обрабатывает взаимодействие с использованием LLM"""
            chain = self.merchant_prompt | self.model
            answer = await chain.ainvoke({
                "conversation_history": "\n".join(self.conversation_history),
                "competitor_sku": self.agent.competitor_sku,
                "current_interaction": str(interaction_data),
                "format_instructions": self.parser.get_format_instructions(),
            })
            print(str(interaction_data),"++++++++++++++++", answer.content,"\n\n")
            return self.parser.parse(answer.content)

        async def step(self) -> None:
            print("WROOOOOOOOOOOOOOOOOOOOOOOOOONG INIT")
            #print("COMPETITOR AGENT STARTED PROCESSING REQUEST")

            # Первое сообщение от MerchantAgent
            request = ShopListRequest.model_validate_json(self.message.content)
            self._update_history("Merchant", f"Initial request: {request.ingredients}")

            # Обрабатываем запрос
            initial_response = await self.process_interaction({
                "type": "initial_request",
                "requested_items": request.ingredients
            })

            # Отправляем ответ
            await self.context.reply_with_inform(self.message).with_content(initial_response.action)
            self._update_history("Competitor", str(initial_response.action))

            # Обрабатываем возможные последующие сообщения в этом диалоге
            while True:
                response = await self.receive(
                    template=MessageTemplate(thread_id=self.context.thread_id),
                    timeout=60
                )

                # Парсим ответ
                try:
                    merchant_response = ShopList.model_validate_json(response.content)
                except ValidationError:
                    merchant_response = Conversate.model_validate_json(response.content)
                #print("____________COMPETITOR",f"{response_type}: {str(merchant_response)}")
                self._update_history("Merchant", f" {str(merchant_response)}")

                # Обрабатываем ответ
                next_action = await self.process_interaction({
                    "type": "merchant_response",
                    "content": merchant_response
                })

                # Отправляем следующий шаг
                await self.context.reply_with_inform(response).with_content(next_action.action)
                self._update_history("Competitor", str(next_action.action))

                # Если получен ShopList - завершаем переговоры
                if isinstance(next_action.action, ShopList):
                    print("!!!!!!!!!!COMPETITOR AGENT FINISHED PROCESSING REQUEST")
                    break



            self.set_is_done()

    def setup(self):
        self.add_behaviour(self.HandleRequestBehaviour(
            config=self.config,
            model=self.default_context.create_chat_model(self.config.model)
        ))
