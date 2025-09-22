import errno
import time
from spade_llm.demo.platform.contractnet.contractnet import *
import os
from collections import UserList
from pathlib import Path
from typing import Optional, Set
from asyncio import sleep as asleep
import asyncio
import aiosqlite
from pydantic import ValidationError
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from spade_llm.core.api import AgentContext, AgentId
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
from spade_llm.demo.platform.contractnet.discovery import AgentDescription, AgentTask, DF_ADDRESS
from spade_llm.core.conf import configuration, Configurable
from spade_llm.demo import models
from spade_llm import consts
from typing import Union, Annotated, List, Tuple, Dict
from spade_llm.demo.platform.contractnet.contractnet import ContractNetResponder, ContractNetRequest, \
    ContractNetProposal, \
    ContractNetResponderBehavior, ContractNetInitiatorBehavior
from spade_llm.demo.platform.contractnet.discovery import AgentDescription, AgentTask, DF_ADDRESS

import logging

logger = logging.getLogger(__name__)

class ShopList(BaseModel):
    """List of ingredients with price"""
    ingredients: Dict[str, int] = Field(
        description="ingredients dict with price",
        default_factory=dict,
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

class ProposalBoard(BaseModel):
    agent: str = Field(description="Name of the agent who proposed")
    proposal: ShopList = Field(description="Current best proposal in ShopList format")
    sku_request: ShopListRequest = Field(description="User sku`s request")
    round_number: int = Field(default=0, description="Текущий номер раунда аукциона")
    total_rounds: int = Field(default=0, description="Всего раундов аукциона")

class ProposalBoardAgentConf(BaseModel):
    total_rounds: int = Field(default=10, description="Общее количество раундов аукциона")
    stable_limit: int = Field(default=3, description="Сколько раундов подряд должна держаться ставка для завершения")

@configuration(ProposalBoardAgentConf)
class ProposalBoardAgent(Agent, Configurable[ProposalBoardAgentConf]):
    proposal_board = ProposalBoard(agent='', proposal=ShopList(), sku_request=ShopListRequest())

    class InitialRequestBehaviour(MessageHandlingBehavior):
        ingredients = [
            "мясо (говядина)", "свёкла", "морковь", "лук репчатый",
            "капуста белокочанная", "картофель", "томатная паста", "чеснок",
            "уксус (лимонный сок)", "лавровый лист", "соль, перец",
            "зелень (укроп/петрушка)", "сметана"
        ]

        def __init__(self, config: ProposalBoardAgentConf):
            super().__init__(MessageTemplate.request())
            self.config = config

        async def step(self) -> None:
            self.agent.proposal_board.sku_request = ShopListRequest(ingredients=self.ingredients)
            await asyncio.sleep(1)

            stable_rounds = 0
            last_winner = None

            for i in range(self.config.total_rounds):
                round_number = i + 1
                logger.info("Starting auction round %d/%d", round_number, self.config.total_rounds)

                request = AuctionContractNetInitiatorBehavior(
                    task=self.agent.proposal_board.sku_request,
                    context=self.context,
                    time_to_wait_for_proposals=10
                )

                self.agent.proposal_board.round_number = round_number
                self.agent.proposal_board.total_rounds = self.config.total_rounds

                self.agent.add_behaviour(request)
                await request.join()

                current_winner = dict(self.agent.proposal_board.proposal.ingredients)

                if last_winner == current_winner:
                    stable_rounds += 1
                    logger.info("Winning bid unchanged for %d rounds", stable_rounds)
                else:
                    stable_rounds = 0
                last_winner = current_winner

                logger.info("Completed auction round %d", round_number)
                logger.info("Current auction state: proposal_board=%s", self.agent.proposal_board)

                requested_ingredients = set(self.agent.proposal_board.sku_request.ingredients)
                proposed_ingredients = set(self.agent.proposal_board.proposal.ingredients.keys())
                all_collected = requested_ingredients.issubset(proposed_ingredients)

                if stable_rounds >= self.config.stable_limit and all_collected:
                    logger.info("Auction finished early at round %d: stable %d rounds and order complete",
                                round_number, self.config.stable_limit)
                    break

            requested_ingredients = set(self.agent.proposal_board.sku_request.ingredients)
            proposed_ingredients = set(self.agent.proposal_board.proposal.ingredients.keys())
            missing_ingredients = requested_ingredients - proposed_ingredients

            if not missing_ingredients:
                total_price = sum(self.agent.proposal_board.proposal.ingredients.values())
                await self.context.reply_with_inform(self.message).with_content(
                    f"Все ингредиенты найдены. Общая цена: {total_price}"
                )
            else:
                await self.context.reply_with_inform(self.message).with_content(
                    f"Не удалось собрать все ингредиенты. Отсутствуют: {', '.join(missing_ingredients)}"
                )

            self.set_is_done()

    class RequestInfoBehaviour(MessageHandlingBehavior):
        def __init__(self, config: ProposalBoardAgentConf):
            super().__init__(MessageTemplate.acknowledge())
            self.config = config

        async def step(self):
            await asyncio.sleep(1)
            await self.context.reply_with_inform(self.message).with_content(self.agent.proposal_board)

    def setup(self):
        self.add_behaviour(self.InitialRequestBehaviour(self.config))
        self.add_behaviour(self.RequestInfoBehaviour(self.config))

class FirstMerchantAgentConf(BaseModel):
    model: str = Field(description="Model name")
    bid_delay: float = Field(default=0, description="Delay between bids")

@configuration(FirstMerchantAgentConf)
class FirstMerchantAgent(Agent, Configurable[FirstMerchantAgentConf]):
    shop_sku = {
        "мясо (говядина)": 350,  # Higher price to encourage competition
        "свёкла": 50,            # Exclusive to FirstMerchant
        "морковь": 35,
        "лук репчатый": 25,
        "капуста белокочанная": 85,
        "картофель": 45,
        "томатная паста": 40,
        "чеснок": 20,
        "уксус (лимонный сок)": 10,  # Exclusive to FirstMerchant
        "лавровый лист": 5,
        "соль, перец": 10,
        "зелень (укроп/петрушка)": 30
    }

    # Rest of the FirstMerchantAgent implementation remains unchanged
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_bid = None

    def setup(self):
        asyncio.create_task(self.register_in_df())
        self.add_behaviour(AuctionBidderBehaviour(config=self.config,
                                                  model=self.default_context.create_chat_model(
                                                      self.config.model)))
        self.add_behaviour(DialogueResponderBehaviour(config=self.config,
                                                      model=self.default_context.create_chat_model(
                                                          self.config.model)))

    async def register_in_df(self):
        context = self.default_context
        await asleep(2)
        await (context.inform(DF_ADDRESS).with_content(self.create_description()))

    def create_description(self) -> AgentDescription:
        return AgentDescription(
            id="first_merchant",
            description="""Агент-магазин который участвует в аукционе""",
            tasks=[
                AgentTask(
                    description="Получение текущего состояния аукциона и запрос на предложение",
                    examples=[
                        """
                        class ShopListRequest(BaseModel):
                            ingredients: List[str] = Field(description="List of ingredients needed")""",
                    ])
            ]
        )

class SecondMerchantAgentConf(BaseModel):
    model: str = Field(description="Model name")
    bid_delay: float = Field(default=1.5, description="Delay between bids")

@configuration(SecondMerchantAgentConf)
class SecondMerchantAgent(Agent, Configurable[SecondMerchantAgentConf]):
    shop_sku = {
        "мясо (говядина)": 320,  # Lower price to undercut FirstMerchant
        "морковь": 30,           # Cheaper than FirstMerchant
        "лук репчатый": 20,      # Cheaper than FirstMerchant
        "капуста белокочанная": 80,
        "картофель": 40,
        "томатная паста": 35,
        "чеснок": 15,
        "лавровый лист": 5,
        "соль, перец": 10,
        "зелень (укроп/петрушка)": 25,
        "сметана": 60            # Exclusive to SecondMerchant
    }

    # Rest of the SecondMerchantAgent implementation remains unchanged
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_bid = None

    def setup(self):
        asyncio.create_task(self.register_in_df())
        self.add_behaviour(AuctionBidderBehaviour(config=self.config,
                                                  model=self.default_context.create_chat_model(
                                                      self.config.model)))
        self.add_behaviour(DialogueResponderBehaviour(config=self.config,
                                                      model=self.default_context.create_chat_model(
                                                          self.config.model)))

    async def register_in_df(self):
        context = self.default_context
        await asleep(3)
        await (context.inform(DF_ADDRESS).with_content(self.create_description()))

    def create_description(self) -> AgentDescription:
        return AgentDescription(
            id="second_merchant",
            description="""Агент-магазин который участвует в аукционе""",
            tasks=[
                AgentTask(
                    description="Получение текущего состояния аукциона и запрос на предложение",
                    examples=[
                        """
                        class ShopListRequest(BaseModel):
                            ingredients: List[str] = Field(description="List of ingredients needed")""",
                    ])
            ]
        )

class AuctionProposal(BaseModel):
    author: AgentId = Field(description="Name of the agent who proposed", default='')
    prop: ShopList = Field(description="proposal_shoplist")

class AuctionContractNetInitiatorBehavior(ContextBehaviour):
    task: ShopListRequest
    proposal: Optional[ShopList] = None
    result: Optional[MessageTemplate]
    time_to_wait_for_proposals: float
    _started_at: float

    def __init__(self,
                 task: ShopListRequest,
                 context: AgentContext,
                 time_to_wait_for_proposals: float = 10):
        super().__init__(context)
        self.time_to_wait_for_proposals = time_to_wait_for_proposals
        self.task = task
        self.result = None

    async def on_start(self) -> None:
        self._started_at = time.time()

    @property
    def is_successful(self) -> bool:
        if self.result and self.result.performative == consts.INFORM:
            return True
        else:
            return False

    async def step(self) -> None:
        agents: List[AgentDescription] = await self.find_agents(self.task)

        if len(agents) > 0:
            proposals: List[ShopList] = await self.get_proposals(agents)
            if len(proposals) > 0:
                self.proposal = await self.extract_winner_and_notify_losers(proposals)
        self.set_is_done()

    async def find_agents(self, task: ShopListRequest) -> List[AgentDescription]:
        await (self.context
               .request(DF_ADDRESS)
               .with_content(AgentSearchRequest(task=task.model_dump_json(), top_k=10)))

        search = await self.receive(MessageTemplate(performative=consts.INFORM, thread_id=self.context.thread_id),
                                    timeout=10)

        if search:
            parsed = AgentSearchResponse.model_validate_json(search.content)
            return parsed.agents
        else:
            return []

    async def get_proposals(self, agents: List[AgentDescription]) -> List[AuctionProposal]:
        sent: Set[str] = set()
        received: Set[str] = set()
        result: List[AuctionProposal] = []

        for agent in agents:
            sent.add(agent.id)
            await (self.context.request_proposal(agent.id).with_content(self.task.model_dump_json()))
        started = time.time()

        deadline = started + self.time_to_wait_for_proposals
        while len(received) < len(sent) and time.time() < deadline:
            response = await self.receive(
                MessageTemplate(performative=consts.PROPOSE, thread_id=self.context.thread_id),
                max(0.1, deadline - time.time()))
            if response:
                received.add(str(response.sender))
                prop = AuctionProposal(author=response.sender, prop=ShopList.model_validate_json(response.content))
                result.append(prop)
        return result

    async def extract_winner_and_notify_losers(self, proposals: List[AuctionProposal]) -> Optional[ShopList]:
        if not proposals:
            return None

        winner = None
        for proposal in proposals:
            if winner is None:
                current_supply = set(self.agent.proposal_board.proposal.ingredients.keys()).intersection(
                    set(self.agent.proposal_board.sku_request.ingredients))

                if set(current_supply) - set(proposal.prop.ingredients.keys()):
                    await self.context.refuse(proposal.author).with_content('')

                elif set(proposal.prop.ingredients.keys()) - set(self.agent.proposal_board.proposal.ingredients.keys()):
                    winner = proposal
                    self.agent.proposal_board.proposal = winner.prop
                    await self.context.accept(proposal.author).with_content('')

                elif sum(proposal.prop.ingredients.values()) < sum(
                        self.agent.proposal_board.proposal.ingredients.values()):
                    winner = proposal
                    self.agent.proposal_board.proposal = winner.prop
                    await self.context.accept(proposal.author).with_content('')

                else:
                    await self.context.refuse(proposal.author).with_content('')
            else:
                await self.context.refuse(proposal.author).with_content('UPDATING')
        if winner is None:
            return self.agent.proposal_board.proposal
        else:
            return winner.prop

class Conversate(BaseModel):
    """Offer to another merchant"""
    offer: str = Field(
        description="деловое предложение конкуренту в формате строки без Markdown"
    )

class Act(BaseModel):
    """Action to perform."""

    action: Union[ShopList, Conversate] = Field(
        description="Action to perform. Должно быть БЕЗ Markdown!!!"
                    "Если ты согласен предложить товары по списку из запроса то верни ShopList с ингредиентами и ценами за которые ты готов их продать. "
                    "Если ты не согласен продать только эти товары то верни Conversate с развернутым и аргументированным предложением другому магазину в деловом стиле почему ты хочешь внести изменение в предложение запросившего."
    )

class Decision(BaseModel):
    decision: Union[ShopList, ShopListRequest] = Field(description="Decision to perform. Должно быть БЕЗ Markdown!!!"
                                                                   "Если ты согласен предложить товары по списку из запроса то верни ShopList с ингредиентами и ценами за которые ты готов их продать. "
                                                                   "Если ты хочешь начать переговоры с другим агентом для совместной ставки то верни ShopListRequest с теми товарами которые тебе нужны.")

class StartDialogueBehaviour(ContextBehaviour):
    def __init__(self, context: AgentContext, config, contragent: str, needed_ingredients: ShopListRequest,
                 model: BaseChatModel):
        super().__init__(context)
        self.config = config
        self.contragent = contragent
        self.needed_ingredients = needed_ingredients
        self.model = model
        self.conversation_history = []

        self.parser = PydanticOutputParser(pydantic_object=Act)
        self.merchant_prompt = ChatPromptTemplate.from_template(
            """Ты - опытный агент магазина, мастер переговоров, стремящийся максимизировать прибыль с изяществом и стратегическим чутьем. Твоя цель - заключить сделку, которая принесет максимальную выгоду, но при этом ты должен оставаться убедительным, профессиональным и гибким. Учти:

            - Каждый раунд переговоров снижает потенциальную прибыль на 20%, поэтому быстрые и эффективные сделки предпочтительны.
            - Ты хочешь продавать товары из своего ассортимента, но готов пойти на разумные уступки ради выгодного сотрудничества.
            - Аукцион принимает ставку, только если она покрывает ВСЕ ингредиенты текущей лучшей ставки и добавляет новые недостающие из запроса клиента ИЛИ предлагает то же покрытие по более низкой цене. Предложение только недостающих ингредиентов без полного покрытия будет отклонено.
            - Если ты уже участвуешь в выигрышной ставке (твои уникальные ингредиенты или их комбинация присутствуют в текущей доске), избегай новых переговоров или ставок, чтобы не ухудшить свою позицию и не потерять прибыль.

            Используй креативные стратегии:
            - Предлагай комбинации товаров, которые увеличивают ценность сделки (например, добавляй популярные товары из своего ассортимента).
            - Если отказываешься, делай это дипломатично, предлагая альтернативу, которая выгодна обеим сторонам.
            - Убеди конкурента, подчеркивая взаимную выгоду, используя логические доводы и, при необходимости, легкий шарм.
            - Если видишь возможность для сотрудничества, предложи совместную ставку, которая усилит позиции обоих агентов.

            Текущий контекст переговоров:
            {conversation_history}

            Ваш ассортимент и цены:
            {shop_sku}

            Текущий запрос/предложение тип: {current_interaction_type}
            Данные: {current_interaction_data}

            Решение:
            1. Если предложение конкурента выгодно и усиливает твою позицию, прими его, вернув ShopList с согласованными товарами и ценами.
            2. Если предложение невыгодно, верни Conversate с ярким, убедительным и профессиональным объяснением, почему ты отклоняешь предложение, и предложи альтернативу (например, добавление твоих товаров или снижение цен для совместной ставки).

            Respond in `json` format\n{format_instructions}. JSON only, without Markdown and additional text. БЕЗ Markdown!!!
            """
        )

    def _update_history(self, role: str, content: str):
        self.conversation_history.append(f"{role}: {content}")
        if len(self.conversation_history) > 5:
            self.conversation_history.pop(0)

    async def process_response(self, interaction_data: dict) -> Act:
        chain = self.merchant_prompt | self.model
        answer = await chain.ainvoke({
            "conversation_history": "\n".join(self.conversation_history),
            "shop_sku": self.agent.shop_sku,
            "current_interaction_type": interaction_data["type"],
            "current_interaction_data": str(interaction_data["data"]),
            "format_instructions": self.parser.get_format_instructions(),
        })
        print(interaction_data["type"], ":", str(interaction_data["data"]), "++++++++++++++++", answer.content, "\n\n")
        return self.parser.parse(answer.content)

    async def step(self):
        thread = await self.context.fork_thread()
        max_iterations = 5
        iteration = 0

        current_offer = self.needed_ingredients

        while iteration < max_iterations:
            if isinstance(current_offer, ShopListRequest):
                await (thread.request(self.contragent).with_content(current_offer.model_dump_json()))
            elif isinstance(current_offer, Conversate):
                await (thread.request(self.contragent).with_content(current_offer.model_dump_json()))

            response = await self.receive(
                template=MessageTemplate(thread_id=thread.thread_id, performative=consts.ACKNOWLEDGE), timeout=60)

            if not response:
                break

            try:
                competitor_response = ShopList.model_validate_json(response.content)
                combined_ingredients = dict(self.agent.shop_sku)
                overlaps = set(combined_ingredients.keys()) & set(competitor_response.ingredients.keys())
                if overlaps:
                    for overlap in overlaps:
                        combined_ingredients[overlap] = min(combined_ingredients[overlap],
                                                            competitor_response.ingredients[overlap])
                combined_ingredients.update(
                    {k: v for k, v in competitor_response.ingredients.items() if k not in self.agent.shop_sku})
                combined_bid = ShopList(ingredients=combined_ingredients)
                self.agent.current_bid = combined_bid
                break
            except ValidationError:
                try:
                    competitor_response = Conversate.model_validate_json(response.content)
                    self._update_history("Opponent", str(competitor_response))
                    act = await self.process_response(
                        {"type": type(competitor_response).__name__, "data": competitor_response.model_dump()})
                    self._update_history("Self", str(act.action))
                    if isinstance(act.action, ShopList):
                        await thread.acknowledge(self.contragent).with_content(act.action.model_dump_json())
                        combined_ingredients = dict(self.agent.shop_sku)
                        overlaps = set(combined_ingredients.keys()) & set(act.action.ingredients.keys())
                        if overlaps:
                            for overlap in overlaps:
                                combined_ingredients[overlap] = min(combined_ingredients[overlap],
                                                                    act.action.ingredients[overlap])
                        combined_ingredients.update(
                            {k: v for k, v in act.action.ingredients.items() if k not in self.agent.shop_sku})
                        combined_bid = ShopList(ingredients=combined_ingredients)
                        self.agent.current_bid = combined_bid
                        break
                    else:
                        current_offer = act.action
                except ValidationError:
                    break

            iteration += 1
        await thread.close()
        self.set_is_done()

class DialogueResponderBehaviour(MessageHandlingBehavior):
    def __init__(self, config, model: BaseChatModel):
        super().__init__(MessageTemplate.request())
        self.config = config
        self.model = model
        self.parser = PydanticOutputParser(pydantic_object=Act)
        self.conversation_history = []
        self.merchant_prompt = ChatPromptTemplate.from_template(
            """Ты - харизматичный и стратегически мыслящий агент магазина, чья цель - максимизировать прибыль через умные и убедительные переговоры. Ты стремишься к взаимовыгодным сделкам, используя дипломатию, креативность и профессиональный подход. Учти:

            - Каждый раунд переговоров снижает потенциальную прибыль на 20%, поэтому быстрые сделки предпочтительны.
            - Ты хочешь продавать свои товары, но готов к компромиссам, если это усилит твою позицию в аукционе.
            - Аукцион принимает ставку, только если она покрывает ВСЕ ингредиенты текущей лучшей ставки и добавляет новые недостающие из запроса клиента ИЛИ предлагает то же покрытие по более низкой цене. Предложение только недостающих ингредиентов без полного покрытия будет отклонено.
            - Если ты уже участвуешь в выигрышной ставке (твои уникальные ингредиенты или их комбинация есть в текущей доске), избегай новых переговоров или ставок, чтобы сохранить прибыль и не ухудшить позицию.

            Стратегии для ярких переговоров:
            - Предлагай привлекательные комбинации товаров, подчеркивая их качество или уникальность (например, "наша свежая зелень идеально дополнит блюдо").
            - Если отклоняешь предложение, делай это вежливо, но с убедительными доводами, предлагая альтернативу, которая выгодна обеим сторонам.
            - Используй логические аргументы и легкий шарм, чтобы убедить конкурента в выгоде сотрудничества.
            - Рассматривай возможность совместной ставки, если она позволит покрыть больше ингредиентов или снизить цену.

            Текущий контекст переговоров:
            {conversation_history}

            Ваш ассортимент и цены:
            {shop_sku}

            Текущий запрос/предложение тип: {current_interaction_type}
            Данные: {current_interaction_data}

            Решение:
            1. Если предложение конкурента выгодно, прими его, вернув ShopList с согласованными товарами и ценами.
            2. Если предложение невыгодно, верни Conversate с ярким, убедительным и профессиональным объяснением отказа, предложив альтернативу, которая усиливает позиции обеих сторон.

            Respond in `json` format\n{format_instructions}. JSON only, without Markdown and additional text. БЕЗ Markdown!!!
            """
        )

    def _update_history(self, role: str, content: str):
        self.conversation_history.append(f"{role}: {content}")
        if len(self.conversation_history) > 5:
            self.conversation_history.pop(0)

    async def process_interaction(self, interaction_data: dict) -> Act:
        chain = self.merchant_prompt | self.model
        answer = await chain.ainvoke({
            "conversation_history": "\n".join(self.conversation_history),
            "shop_sku": self.agent.shop_sku,
            "current_interaction_type": interaction_data["type"],
            "current_interaction_data": str(interaction_data["data"]),
            "format_instructions": self.parser.get_format_instructions(),
        })
        print(interaction_data["type"], ":", str(interaction_data["data"]), "++++++++++++++++", answer.content, "\n\n")
        return self.parser.parse(answer.content)

    async def step(self):
        try:
            current_interaction = ShopListRequest.model_validate_json(self.message.content)
        except ValidationError:
            try:
                current_interaction = Conversate.model_validate_json(self.message.content)
            except ValidationError:
                await self.context.reply_with_refuse(self.message).with_content("Invalid request format")
                return

        self._update_history("Opponent", str(current_interaction))

        act = await self.process_interaction(
            {"type": type(current_interaction).__name__, "data": current_interaction.model_dump()})

        self._update_history("Self", str(act.action))

        if isinstance(act.action, ShopList):
            await self.context.reply_with_acknowledge(self.message).with_content(act.action.model_dump_json())
        elif isinstance(act.action, Conversate):
            await self.context.reply_with_acknowledge(self.message).with_content(act.action.model_dump_json())
        else:
            await self.context.reply_with_refuse(self.message).with_content("Unexpected action type")

class AuctionBidderBehaviour(MessageHandlingBehavior):
    def __init__(self, config, model: BaseChatModel):
        super().__init__(MessageTemplate.request_proposal())
        self.config = config
        self.model = model

        self.decision_parser = PydanticOutputParser(pydantic_object=Decision)
        self.shoplist_parser = PydanticOutputParser(pydantic_object=ShopList)
        self.merchant_prompt = ChatPromptTemplate.from_template(
            """Ты - амбициозный агент магазина, мастер аукционов, стремящийся доминировать в торгах, максимизируя прибыль через стратегические и яркие ходы. Твоя цель - выиграть аукцион, предложив лучшую ставку или заключив выгодное сотрудничество. Если ты не участвуешь в победной ставке, ты не получишь прибыль. Если текущая лучшая ставка не покрывает все ингредиенты, никто не выигрывает, но ты можешь сделать неполную ставку, чтобы привлечь партнеров. Учти:

            - Новая ставка принимается, только если она:
              - Покрывает ВСЕ ингредиенты текущей лучшей ставки.
              - Добавляет новые недостающие ингредиенты из запроса клиента.
              - Или предлагает то же покрытие по более низкой цене.
            - Предложение только недостающих ингредиентов без полного покрытия текущей ставки будет отклонено.
            - Если ты уже участвуешь в выигрышной ставке (твои уникальные ингредиенты или их комбинация есть в текущей доске), не делай новых ставок или переговоров, чтобы сохранить прибыль.

            Стратегии для ярких и прибыльных торгов:
            - Анализируй текущую доску и предлагай комбинации товаров, которые усиливают твое предложение (например, добавляй популярные товары или снижай цену на ключевые ингредиенты для конкурентного преимущества).
            - Если не можешь покрыть все ингредиенты, начни переговоры для совместной ставки, убедительно подчеркивая взаимную выгоду.
            - Используй креативные подходы: предлагай скидки на определенные товары, если клиент берет полный комплект, или подчеркивай уникальность твоего ассортимента.
            - Если видишь, что конкурент близок к победе, предложи сотрудничество, чтобы разделить прибыль, а не терять все.

            Ваш ассортимент и цены:
            {my_sku}

            Текущее лидирующее предложение (текущая лучшая ставка, которую нужно улучшить):
            {current_board}

            Список доступных к переговорам агентов-магазинов:
            {agents_able_to_conversate}

            Решение:
            1. Если можешь предложить конкурентоспособную ставку, верни ShopList с ингредиентами и ценами, покрывающими текущую ставку и добавляющими новые или снижающими цену.
            2. Если нужна совместная ставка, верни ShopListRequest с товарами, которые тебе нужны от партнера, чтобы покрыть запрос клиента.

            Respond in `json` format\n{format_instructions}. JSON only, without Markdown and additional text. БЕЗ Markdown!!!
            """
        )

    async def get_current_board(self) -> ProposalBoard:
        await asleep(self.config.bid_delay)
        await self.context.acknowledge('proposal_board').with_content('')
        response = await self.receive(
            template=MessageTemplate(thread_id=self.context.thread_id, performative=consts.INFORM),
            timeout=60,
        )
        board = ProposalBoard.model_validate_json(response.content)
        return board

    async def update_my_bid(self):
        current_board = await self.get_current_board()

        agents_able = ["second_merchant"] if self.context.agent_type == 'first_merchant' else ["first_merchant"]

        chain = self.merchant_prompt | self.model
        answer = await chain.ainvoke({
            "my_sku": self.agent.shop_sku,
            "current_board": current_board,
            "format_instructions": self.decision_parser.get_format_instructions(),
            "agents_able_to_conversate": agents_able
        })

        try:
            response = self.decision_parser.parse(answer.content).decision
        except Exception as e:
            logger.error("Ошибка при парсинге ответа decision_parser: %s", e)
            response = ShopList(ingredients={})

        if isinstance(response, ShopList):
            self.agent.current_bid = response
        else:
            if self.context.agent_type == 'first_merchant':
                contragent_type = 'second_merchant'
            else:
                contragent_type = 'first_merchant'

            start_conversation = StartDialogueBehaviour(
                self.context, self.config, contragent_type, response, self.model
            )
            self.agent.add_behaviour(start_conversation)
            await start_conversation.join()

            if self.agent.current_bid is None:
                chain = self.merchant_prompt | self.model
                fallback_answer = await chain.ainvoke({
                    "my_sku": self.agent.shop_sku,
                    "current_board": current_board,
                    "format_instructions": self.shoplist_parser.get_format_instructions(),
                    "agents_able_to_conversate": agents_able
                })
                self.agent.current_bid = self.shoplist_parser.parse(fallback_answer.content)

        return

    async def get_my_bid(self):
        if self.agent.current_bid is None:
            await self.update_my_bid()
        return self.agent.current_bid.model_dump_json()

    async def step(self) -> None:
        my_bid = await self.get_my_bid()

        await asyncio.sleep(self.config.bid_delay)
        await self.context.reply_with_propose(self.message).with_content(my_bid)

        response = await self.receive(MessageTemplate(thread_id=self.context.thread_id), timeout=15)

        if response.content == 'UPDATING':
            print('UPDATING')
        else:
            if response.performative == consts.ACCEPT:
                pass
            elif response.performative == consts.REFUSE:
                await self.update_my_bid()
            else:
                pass