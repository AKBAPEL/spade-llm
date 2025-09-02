import errno
import logging
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


class ProposalBoardAgentConf(BaseModel):
    pass


@configuration(ProposalBoardAgentConf)
class ProposalBoardAgent(Agent, Configurable[ProposalBoardAgentConf]):
    proposal_board = ProposalBoard(agent='', proposal=ShopList(), sku_request=ShopListRequest())

    class InitialRequestBehaviour(MessageHandlingBehavior):
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

        def __init__(self, config: ProposalBoardAgentConf):
            super().__init__(MessageTemplate.request())
            self.config = config

        async def step(self) -> None:
            # Create request model
            self.agent.proposal_board.sku_request = ShopListRequest(
                ingredients=self.ingredients)  # Тут в будущем будет составляться список. Пока хардкодим
            print(777, self.agent.proposal_board.sku_request.model_dump_json())
            await asyncio.sleep(1)
            for i in range(3):
                request = AuctionContractNetInitiatorBehavior(
                    task=self.agent.proposal_board.sku_request,
                    context=self.context,
                    time_to_wait_for_proposals=10
                )

                self.agent.add_behaviour(request)
                await request.join()
            await self.context.reply_with_inform(self.message).with_content('Все ок')
            self.set_is_done()

    class RequestInfoBehaviour(MessageHandlingBehavior):
        def __init__(self, config: ProposalBoardAgentConf):
            super().__init__(MessageTemplate.acknowledge())
            self.config = config

        async def step(self):
            logger.info("Sent board status to %s", self.message.sender.agent_type)
            await asyncio.sleep(1)
            await self.context.reply_with_inform(self.message).with_content(self.agent.proposal_board)

    def setup(self):
        self.add_behaviour(self.InitialRequestBehaviour(self.config))
        self.add_behaviour(self.RequestInfoBehaviour(self.config))
        # self.add_behaviour(self.HandleProposalBehaviour(self.config))


# --- //// --- Merchants --- //// --- --- /// ---
class FirstMerchantAgentConf(BaseModel):
    model: str = Field(description="Model name")
    bid_delay: float = Field(default=0, description="Delay between bids")


@configuration(FirstMerchantAgentConf)
class FirstMerchantAgent(Agent, Configurable[FirstMerchantAgentConf]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_bid = None

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

    def setup(self):
        asyncio.create_task(self.register_in_df())
        self.add_behaviour(AuctionBidderBehaviour(config=self.config,
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_bid = None

    shop_sku = {
        "мясо (говядина)": 320,
        # "свёкла": 50,
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
        # }
    }

    def setup(self):
        asyncio.create_task(self.register_in_df())
        self.add_behaviour(AuctionBidderBehaviour(config=self.config,
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


# ---- AUCTION CONTRACT NET ----
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
        logger.info("Found %i agents for task", len(agents))

        if len(agents) > 0:
            proposals: List[ShopList] = await self.get_proposals(agents)
            logger.info("Got %i proposals for task", len(proposals))
            if len(proposals) > 0:
                self.proposal = await self.extract_winner_and_notify_losers(proposals)
                logger.info("Best proposal is %s", self.proposal)
            else:
                logger.error("Failed to get any proposals in time")
        else:
            logger.error("Failed to find any agents")
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
            logger.error(f"No search received in {10} seconds")
            return []

    async def get_proposals(self, agents: List[AgentDescription]) -> List[AuctionProposal]:
        sent: Set[str] = set()
        received: Set[str] = set()
        result: List[AuctionProposal] = []

        # request = ContractNetRequest(task=self.task.model_dump_json())
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
                # Tyt если в один момент два пропозала то может не считать
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
                logger.info("В предложении агента нет необходимых ингредиентов: %s",
                            str(set(current_supply) - set(proposal.prop.ingredients.keys())))

                # Check if proposal is missing any ingredients from the current best
                if set(current_supply) - set(proposal.prop.ingredients.keys()):
                    # ТУТ НА САМОМ ДЕЛЕ ОТКАЗ ТОЛЬКО ЕСЛИ НЕТ КАКОГО ТО ИЗ ЗАПРАШИВАЕМЫХ КОТОРЫЕ ЕСТЬ В ПРЕДЛОЖЕНИИ  А НЕ ИЗ СТАВКИ
                    logger.info('Missing some ingredients from the current best. Refusing')
                    await self.context.refuse(proposal.author).with_content('')

                # Check if proposal has all ingredients plus extra from request
                elif set(proposal.prop.ingredients.keys()) - set(self.agent.proposal_board.proposal.ingredients.keys()):
                    logger.info('Has all ingredients plus extra from request. Accepting')
                    winner = proposal
                    self.agent.proposal_board.proposal = winner.prop
                    await self.context.accept(proposal.author).with_content('')

                # Compare prices if both proposals have the same ingredients
                elif sum(proposal.prop.ingredients.values()) < sum(
                        self.agent.proposal_board.proposal.ingredients.values()):
                    logger.info('Lower price for same ingredients. Accepting')
                    winner = proposal
                    self.agent.proposal_board.proposal = winner.prop
                    await self.context.accept(proposal.author).with_content('')

                else:
                    logger.info('Higher or equal price, rejecting. Refusing')
                    await self.context.refuse(proposal.author).with_content('')
            else:
                logger.info('Already updated board. Refusing')
                # ТУТ ЕСЛИ ВОЗВРАЩАЕМ UPDATING то агенту надо заретраить посылку
                await self.context.refuse(proposal.author).with_content('UPDATING')
        # Update the proposal board with the winner
        # self.agent.proposal_board.proposal = winner
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
        description="Action to perform. "
                    "Если ты согласен предложить товары по списку из запроса то верни ShopList с ингредиентами и ценами за которые ты готов их продать. "
                    "Если ты не согласен продать только эти товары то верни Conversate с развернутым и аргументированным предложением другому магазину в деловом стиле почему ты хочешь внести изменение в предложение запросившего."

    )


class Decision(BaseModel):
    decision: Union[ShopList, str] = Field(description="Decision to perform."
                                                       "Если ты согласен предложить товары по списку из запроса то верни ShopList с ингредиентами и ценами за которые ты готов их продать. "
                                                       "Если ты хочешь начать переговоры с другим агентом для совместной ставки то верни строку StartDialogue")


class AuctionBidderBehaviour(MessageHandlingBehavior):
    def __init__(self, config, model: BaseChatModel):
        super().__init__(MessageTemplate.request_proposal())
        self.config = config
        self.model = model

        self.decision_parser = PydanticOutputParser(pydantic_object=Decision)
        self.shoplist_parser = PydanticOutputParser(pydantic_object=ShopList)
        self.merchant_prompt = ChatPromptTemplate.from_template(
            """Ты - агент магазина. Ваша цель - максимизировать прибыль. Если ты не участвуешь в победной ставке то ты не получаешь прибыль. 
            Если по результатом аукциона в текущей лучшей ставке нет всех ингредиентов из запроса, то никто не получает прибыль.
            Но тебе никто не запрещает делать неполную ставку. Если ее примут, то другие агенты могут начать с тобой сотрудничать.

            Ваш ассортимент и цены:
            {my_sku}
            
            Текущее лидирующее предложение:
            {current_board}
            
            Список доступных к перегвоорам агентов-магазинов, которые участвуют в аукционе (если их нет то ты не можешь начать переговоры): 
            {agents_able_to_conversate}
            
            Тебе нужно взвесить все за и против и принять решение в данной ситуации:
                1. Отправлять предложение самому. Тогда тебе нужно понять - какие ингредиенты и по какой цене ты хочешь предложить на этой итерации аукциона в формате ShopList.
                2. Начать переговоры с другим агентом-магазином, чтобы сделать совместную ставку на аукционе. Тогда верни строку StartDialogue.
            Respond in `json` format\n{format_instructions}. JSON only, without Markdown and additional text"""
        )

    async def get_current_board(self) -> ProposalBoard:
        await asleep(self.config.bid_delay)
        await self.context.acknowledge('proposal_board').with_content('')
        response = await self.receive(
            template=MessageTemplate(thread_id=self.context.thread_id, performative=consts.INFORM),
            timeout=60,
        )
        return ProposalBoard.model_validate_json(response.content)

    async def update_my_bid(self):
        current_board = await self.get_current_board()

        chain = self.merchant_prompt | self.model
        answer = await chain.ainvoke({
            "my_sku": self.agent.shop_sku,
            "current_board": current_board,
            "format_instructions": self.decision_parser.get_format_instructions(),
            "agents_able_to_conversate": []
        })
        response = self.decision_parser.parse(answer.content).decision
        print(8888888888888888888888888, response)
        # Просто заглушка отдаем все  ингредиенты
        # self.agent.current_bid = ShopList(ingredients={key: self.agent.shop_sku[key] for key in current_board.sku_request.ingredients if
        #                                key in self.agent.shop_sku.keys()})
        #self.agent.current_bid = self.shoplist_parser.parse(response)
        if isinstance(response, ShopList):
            self.agent.current_bid = response
            print("+++", self.agent.current_bid)

        else:
            start_conversation = StartDialogueBehaviour()
            # Тут надо еще много допилить
            self.agent.add_behaviour(start_conversation)
            await start_conversation.join()
        return
    async def get_my_bid(self):
        if self.agent.current_bid is None:
            await self.update_my_bid()
        # надо добавить логику того что агент может отказаться от ставки
        return self.agent.current_bid.model_dump_json()

    async def step(self) -> None:
        my_bid = await self.get_my_bid()

        await asyncio.sleep(self.config.bid_delay)  # небольшой await чтобы не было коллизий по отправке в один момент
        await self.context.reply_with_propose(self.message).with_content(my_bid)

        response = await self.receive(MessageTemplate(thread_id=self.context.thread_id), timeout=15)

        if response.content == 'UPDATING':
            # тут надо ничего не делать, оставить текущую ставку
            print('UPDATING')
        else:
            # тут надо свитч на аксепт или рефуз
            if response.performative == consts.ACCEPT:
                # тут надо обозначить что моя текущая ставка - лидирующая
                pass
            elif response.performative == consts.REFUSE:
                # если отказали то надо обновить предложение
                await self.update_my_bid()
            else:
                # ??
                pass
