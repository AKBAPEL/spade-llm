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
            request = AuctionContractNetInitiatorBehavior(
                task=self.agent.proposal_board.sku_request,
                context=self.context,
                time_to_wait_for_proposals=10
            )

            self.agent.add_behaviour(request)
            await request.join()
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
            print('INITED REQUEST')

        async def step(self):
            print('-----------------', self.message.sender)
            await self.context.reply_with_inform(self.message).with_content(self.agent.proposal_board)

    def setup(self):
        self.add_behaviour(self.InitialRequestBehaviour(self.config))
        self.add_behaviour(self.RequestInfoBehaviour(self.config))
        # self.add_behaviour(self.HandleProposalBehaviour(self.config))


# --- //// --- Merchants --- //// --- --- /// ---

class FirstMerchantAgentConf(BaseModel):
    model: str = Field(description="Model name")


@configuration(FirstMerchantAgentConf)
class FirstMerchantAgent(Agent, Configurable[FirstMerchantAgentConf]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

    class HandleRequestProposalBehaviour(MessageHandlingBehavior):

        def __init__(self, config: FirstMerchantAgentConf, model: BaseChatModel):
            super().__init__(MessageTemplate.request_proposal())
            self.config = config
            self.model = model

        async def get_current_board(self) -> ProposalBoard:
            await self.context.acknowledge('proposal_board').with_content('')
            response = await self.receive(
                template=MessageTemplate(thread_id=self.context.thread_id, performative=consts.INFORM),
                timeout=60,
            )
            return ProposalBoard.model_validate_json(response.content)

        async def step(self) -> None:
            print(123)
            # Parse incoming request
            request = ShopListRequest.model_validate_json(self.message.content)
            # print(71313131313, request)
            my_offer = ShopList(ingredients={key: self.agent.shop_sku[key] for key in request.ingredients if
                                             key in self.agent.shop_sku.keys()})
            await asyncio.sleep(0.11)
            await self.context.reply_with_propose(self.message).with_content(my_offer)

    def setup(self):
        asyncio.create_task(self.register_in_df())
        self.add_behaviour(self.HandleRequestProposalBehaviour(config=self.config,
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


@configuration(SecondMerchantAgentConf)
class SecondMerchantAgent(Agent, Configurable[SecondMerchantAgentConf]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

    class HandleRequestProposalBehaviour(MessageHandlingBehavior):

        def __init__(self, config: FirstMerchantAgentConf, model: BaseChatModel):
            super().__init__(MessageTemplate.request_proposal())
            self.config = config
            self.model = model

        async def get_current_board(self) -> ProposalBoard:  # надо использовать !!
            print(123)
            await self.context.acknowledge('proposal_board').with_content('')
            response = await self.receive(
                template=MessageTemplate(thread_id=self.context.thread_id, performative=consts.INFORM),
                timeout=60,
            )
            return ProposalBoard.model_validate_json(response.content)

        async def step(self) -> None:
            # Parse incoming request
            request = ShopListRequest.model_validate_json(self.message.content)
            # print(71313131313, request)
            my_offer = ShopList(ingredients={key: self.agent.shop_sku[key] for key in request.ingredients if
                                             key in self.agent.shop_sku.keys()})
            await self.context.reply_with_propose(self.message).with_content(my_offer)

    def setup(self):
        asyncio.create_task(self.register_in_df())
        self.add_behaviour(self.HandleRequestProposalBehaviour(config=self.config,
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
                # Check if proposal is missing any ingredients from the current best
                if set(self.agent.proposal_board.proposal.ingredients.keys()) - set(proposal.prop.ingredients.keys()):
                    logger.info('Missing some ingredients from the current best')
                    await self.context.refuse(proposal.author).with_content('')
                # Check if proposal has all ingredients plus extra from request
                elif set(proposal.prop.ingredients.keys()) - set(self.agent.proposal_board.proposal.ingredients.keys()):
                    logger.info('Has all ingredients plus extra from request')
                    winner = proposal
                    self.agent.proposal_board.proposal = winner.prop
                    await self.context.accept(proposal.author).with_content('')
                # Compare prices if both proposals have the same ingredients
                elif sum(proposal.prop.ingredients.values()) < sum(
                        self.agent.proposal_board.proposal.ingredients.values()):
                    logger.info('Lower price for same ingredients')
                    winner = proposal
                    self.agent.proposal_board.proposal = winner.prop
                    await self.context.accept(proposal.author).with_content('')
                else:
                    logger.info('Higher or equal price, rejecting')
                    await self.context.refuse(proposal.author).with_content('')
            else:
                logger.info('Already updated board, rejecting')
                # ТУТ ЕСЛИ ВОЗВРАЩАЕМ UPDATING то агенту надо заретраить посылку
                await self.context.refuse(proposal.author).with_content('UPDATING')
        # Update the proposal board with the winner
        # self.agent.proposal_board.proposal = winner
        if winner is None:
            return self.agent.proposal_board.proposal
        else:
            return winner.prop
