import asyncio
import logging
import json
from typing import Dict, List, Optional, Set, Union

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, ValidationError

from spade_llm import consts
from spade_llm.core.agent import Agent
from spade_llm.core.api import AgentContext, AgentId
from spade_llm.core.behaviors import ContextBehaviour, MessageHandlingBehavior, MessageTemplate
from spade_llm.core.conf import Configurable, configuration
from spade_llm.demo.platform.contractnet.contractnet import *
from spade_llm.demo.platform.contractnet.discovery import AgentDescription, AgentSearchRequest, AgentTask, DF_ADDRESS
from spade_llm.demo.platform.auction.agent_prompts import *

logger = logging.getLogger(__name__)


# Models for auction data
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
    total_rounds: int = Field(default=6, description="Общее количество раундов аукциона")
    stable_limit: int = Field(default=3, description="Сколько раундов подряд должна держаться ставка для завершения")


@configuration(ProposalBoardAgentConf)
class ProposalBoardAgent(Agent, Configurable[ProposalBoardAgentConf]):
    """Agent managing the auction proposal board"""
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
            """Run the auction process for the requested ingredients"""
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
                await asyncio.sleep(3)
                current_winner = dict(self.agent.proposal_board.proposal)

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
            """Reply with the current state of the proposal board"""
            await asyncio.sleep(1)
            await self.context.reply_with_inform(self.message).with_content(self.agent.proposal_board)

    def setup(self):
        """Initialize agent behaviors"""
        self.add_behaviour(self.InitialRequestBehaviour(self.config))
        self.add_behaviour(self.RequestInfoBehaviour(self.config))


class FirstMerchantAgentConf(BaseModel):
    model: str = Field(description="Model name")
    bid_delay: float = Field(default=0, description="Delay between bids")


@configuration(FirstMerchantAgentConf)
class FirstMerchantAgent(Agent, Configurable[FirstMerchantAgentConf]):
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
        "зелень (укроп/петрушка)": 30
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_bid = None

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
        await asyncio.sleep(2)
        await context.inform(DF_ADDRESS).with_content(self.create_description())

    def create_description(self) -> AgentDescription:
        """Create a description for the agent"""
        return AgentDescription(
            id="first_merchant",
            description="""Агент-магазин который участвует в аукционе""",
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


class SecondMerchantAgentConf(BaseModel):
    model: str = Field(description="Model name")
    bid_delay: float = Field(default=1.5, description="Delay between bids")


@configuration(SecondMerchantAgentConf)
class SecondMerchantAgent(Agent, Configurable[SecondMerchantAgentConf]):
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_bid = None

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
        await asyncio.sleep(3)
        await context.inform(DF_ADDRESS).with_content(self.create_description())

    def create_description(self) -> AgentDescription:
        """Create a description for the agent"""
        return AgentDescription(
            id="second_merchant",
            description="""Агент-магазин который участвует в аукционе""",
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


class AuctionProposal(BaseModel):
    author: AgentId = Field(description="Name of the agent who proposed", default='')
    prop: ShopList = Field(description="proposal_shoplist")


class AuctionContractNetInitiatorBehavior(ContextBehaviour):
    """Behavior for initiating the contract net protocol in the auction"""
    task: ShopListRequest
    proposal: Optional[ShopList] = None
    result: Optional[MessageTemplate]
    time_to_wait_for_proposals: float
    _started_at: float

    def __init__(self, task: ShopListRequest, context: AgentContext, time_to_wait_for_proposals: float = 10):
        super().__init__(context)
        self.time_to_wait_for_proposals = time_to_wait_for_proposals
        self.task = task
        self.result = None

    async def on_start(self) -> None:
        """Initialize the start time for the behavior"""
        self._started_at = time.time()

    @property
    def is_successful(self) -> bool:
        """Check if the behavior completed successfully"""
        return bool(self.result and self.result.performative == consts.INFORM)

    async def step(self) -> None:
        """Execute the auction contract net protocol"""
        agents = await self.find_agents(self.task)
        if agents:
            proposals = await self.get_proposals(agents)
            if proposals:
                self.proposal = await self.extract_winner_and_notify_losers(proposals)
        self.set_is_done()

    async def find_agents(self, task: ShopListRequest) -> List[AgentDescription]:
        """Find agents capable of fulfilling the task"""
        await self.context.request(DF_ADDRESS).with_content(AgentSearchRequest(task=task.model_dump_json(), top_k=10))
        search = await self.receive(
            MessageTemplate(performative=consts.INFORM, thread_id=self.context.thread_id),
            timeout=10
        )
        if search:
            parsed = AgentSearchResponse.model_validate_json(search.content)
            return parsed.agents
        return []

    async def get_proposals(self, agents: List[AgentDescription]) -> List[AuctionProposal]:
        """Collect proposals from agents"""
        sent: Set[str] = set()
        received: Set[str] = set()
        result: List[AuctionProposal] = []

        for agent in agents:
            sent.add(agent.id)
            await self.context.request_proposal(agent.id).with_content(self.task.model_dump_json())

        started = time.time()
        deadline = started + self.time_to_wait_for_proposals
        while len(received) < len(sent) and time.time() < deadline:
            response = await self.receive(
                MessageTemplate(performative=consts.PROPOSE, thread_id=self.context.thread_id),
                max(0.1, deadline - time.time())
            )
            if response:
                received.add(str(response.sender))
                prop = AuctionProposal(author=response.sender, prop=ShopList.model_validate_json(response.content))
                result.append(prop)
        return result

    async def extract_winner_and_notify_losers(self, proposals: List[AuctionProposal]) -> Optional[ShopList]:
        """Select the winning proposal and notify losers"""
        if not proposals:
            return None

        winner = None
        for proposal in proposals:
            if winner is None:
                current_supply = set(self.agent.proposal_board.proposal.ingredients.keys()).intersection(
                    set(self.agent.proposal_board.sku_request.ingredients)
                )

                if set(current_supply) - set(proposal.prop.ingredients.keys()):
                    logger.info("Missing ingredients from current best. Refusing")
                    await self.context.refuse(proposal.author).with_content('')
                elif set(proposal.prop.ingredients.keys()) - set(self.agent.proposal_board.proposal.ingredients.keys()):
                    logger.info("Has all ingredients plus extra from request. Accepting")
                    winner = proposal
                    self.agent.proposal_board.proposal = winner.prop
                    await self.context.accept(proposal.author).with_content('')
                elif sum(proposal.prop.ingredients.values()) < sum(
                        self.agent.proposal_board.proposal.ingredients.values()):
                    logger.info("Lower price for same ingredients. Accepting")
                    winner = proposal
                    self.agent.proposal_board.proposal = winner.prop
                    await self.context.accept(proposal.author).with_content('')
                else:
                    logger.info("Higher or equal price, rejecting")
                    await self.context.refuse(proposal.author).with_content('')
            else:
                logger.info("Already updated board. Refusing")
                await self.context.refuse(proposal.author).with_content('UPDATING')
        return winner.prop if winner else self.agent.proposal_board.proposal


class Conversate(BaseModel):
    """Offer to another merchant"""
    offer: str = Field(
        description="деловое предложение конкуренту в формате строки без Markdown"
    )


class Act(BaseModel):
    """Action to perform."""
    action: Union[ShopList, Conversate] = Field(
        description="Action to perform. Должно быть БЕЗ Markdown!!! "
                    "Если ты согласен предложить товары по списку из запроса то верни ShopList с ингредиентами и ценами за которые ты готов их продать. "
                    "Если ты не согласен продать только эти товары то верни Conversate с развернутым и аргументированным предложением другому магазину в деловом стиле почему ты хочешь внести изменение в предложение запросившего."
    )


class CollaborationProposal(BaseModel):
    """Proposal for collaboration with another agent"""
    target_agent: str = Field(
        description="agent_type агента к которому будет направлено предложение о сотрудничестве"
    )
    needed_ingredients: ShopList = Field(
        description="ShopList с теми товарами и ценами которые тебе нужны для совместной ставки"
    )


class Decision(BaseModel):
    """Decision to perform."""
    decision: Union[ShopList, CollaborationProposal] = Field(
        description="Decision to perform. Должно быть БЕЗ Markdown!!! "
                    "Если ты согласен предложить товары по списку из запроса то верни ShopList с ингредиентами и ценами за которые ты готов их продать. "
                    "Если ты хочешь начать переговоры с другим агентом для совместной ставки то верни CollaborationProposal с target_agent (agent_type партнера) и needed_ingredients (ShopList с теми товарами и ценами которые тебе нужны)."
    )


class StartDialogueBehaviour(ContextBehaviour):
    """Behavior for initiating a dialogue with another merchant"""

    def __init__(self, context: AgentContext, config, contragent: str, needed_ingredients: ShopList,
                 model: BaseChatModel):
        super().__init__(context)
        self.config = config
        self.contragent = contragent
        self.needed_ingredients = needed_ingredients
        self.model = model
        self.conversation_history = []
        self.parser = PydanticOutputParser(pydantic_object=Act)
        self.shoplist_parser = PydanticOutputParser(pydantic_object=ShopList)
        self.conversate_parser = PydanticOutputParser(pydantic_object=Conversate)
        self.merchant_prompt = DIALOGUE_INITIATOR_PROMPT

    def _update_history(self, role: str, content: Union[ShopList, Conversate]):
        """Update conversation history with new message"""
        self.conversation_history.append(f"{role}: {content.model_dump_json()}")
        if len(self.conversation_history) > 5:
            self.conversation_history.pop(0)

    def _save_dialogue(self):
        """Save the conversation history to a file"""
        import os
        import json
        os.makedirs("dialogues", exist_ok=True)
        filename = f"dialogue_start_{time.time()}.txt"
        path = os.path.join("dialogues", filename)

        def parse_saved_content(content: str) -> tuple[str, str]:
            """Parse content to determine its type and format for saving"""
            try:
                data = json.loads(content)
                if 'offer' in data:
                    try:
                        conversate = Conversate.model_validate(data)
                        return "Conversate", conversate.offer
                    except ValidationError:
                        pass
                if 'ingredients' in data:
                    try:
                        shop_list = ShopList.model_validate(data)
                        return "ShopList", str(shop_list.ingredients)
                    except ValidationError:
                        pass
            except json.JSONDecodeError:
                pass
            return "Unknown", content

        with open(path, "w", encoding="utf-8") as f:
            for entry in self.conversation_history:
                try:
                    role, content = entry.split(": ", 1)
                    msg_type, display_content = parse_saved_content(content)
                    f.write(f"{role} ({msg_type}): {display_content}\n")
                except Exception:
                    f.write(f"ParseError: {entry}\n")

        logger.info("Saved dialogue to %s (lines: %d)", path, len(self.conversation_history))

    async def process_response(self, interaction_data: dict) -> Act:
        """Process incoming interaction and generate a response"""

        def clean_json(text: str) -> str:
            text = text.strip()
            if text.startswith("```json\n"):
                text = text[7:]
            if text.endswith("```"):
                text = text[:-3]
            return text.strip()

        chain = self.merchant_prompt | self.model
        answer = await chain.ainvoke({
            "conversation_history": "\n".join(self.conversation_history),
            "shop_sku": self.agent.shop_sku,
            "current_interaction_type": interaction_data["type"],
            "current_interaction_data": str(interaction_data["data"]),
            "format_instructions": self.parser.get_format_instructions(),
        })
        cleaned_content = clean_json(answer.content)
        return self.parser.parse(cleaned_content)

    async def step(self):
        """Execute the dialogue process"""
        logger.info("Starting dialogue with %s", self.contragent)
        thread = await self.context.fork_thread()
        max_iterations = 10
        iteration = 0
        self._update_history("Self", self.needed_ingredients)
        current_offer = self.needed_ingredients

        while iteration < max_iterations:
            if isinstance(current_offer, (ShopList, Conversate)):
                await thread.request(self.contragent).with_content(current_offer.model_dump_json())
            else:
                logger.warning("Unexpected offer type: %s", type(current_offer))
                break

            response = await self.receive(
                template=MessageTemplate(thread_id=thread.thread_id, performative=consts.ACKNOWLEDGE),
                timeout=60
            )
            # TODO: check if response is Refuse and break
            if not response:
                logger.warning("No response from %s", self.contragent)
                break

            try:
                content_dict = json.loads(response.content)
                if "ingredients" in content_dict:
                    competitor_msg = ShopList.model_validate_json(response.content)
                    msg_type = "ShopList"
                elif "offer" in content_dict:
                    competitor_msg = Conversate.model_validate_json(response.content)
                    msg_type = "Conversate"
                else:
                    logger.error("Unrecognized message format from %s: %s", self.contragent, response.content)
                    break
            except json.JSONDecodeError:
                logger.error("Invalid JSON format from %s: %s", self.contragent, response.content)
                break

            self._update_history("Opponent", competitor_msg)
            if isinstance(competitor_msg, ShopList):
                combined_ingredients = dict(self.agent.shop_sku)
                overlaps = set(combined_ingredients.keys()) & set(competitor_msg.ingredients.keys())
                for overlap in overlaps:
                    combined_ingredients[overlap] = min(combined_ingredients[overlap],
                                                        competitor_msg.ingredients[overlap])
                combined_ingredients.update(
                    {k: v for k, v in competitor_msg.ingredients.items() if k not in self.agent.shop_sku}
                )
                self.agent.current_bid = ShopList(ingredients=combined_ingredients)
                break
            elif isinstance(competitor_msg, Conversate):
                act = await self.process_response({
                    "type": msg_type,
                    "data": competitor_msg.model_dump()
                })
                self._update_history("Self", act.action)
                if isinstance(act.action, ShopList):
                    await thread.acknowledge(self.contragent).with_content(act.action.model_dump_json())
                    combined_ingredients = dict(self.agent.shop_sku)
                    overlaps = set(combined_ingredients.keys()) & set(act.action.ingredients.keys())
                    for overlap in overlaps:
                        combined_ingredients[overlap] = min(combined_ingredients[overlap],
                                                            act.action.ingredients[overlap])
                    combined_ingredients.update(
                        {k: v for k, v in act.action.ingredients.items() if k not in self.agent.shop_sku}
                    )
                    self.agent.current_bid = ShopList(ingredients=combined_ingredients)
                    break
                elif isinstance(act.action, Conversate):
                    current_offer = act.action
                else:
                    logger.warning("Unexpected action type: %s", type(act.action))
                    break
            iteration += 1

        await thread.close()
        self._save_dialogue()
        self.set_is_done()


class DialogueResponderBehaviour(MessageHandlingBehavior):
    """Behavior for responding to dialogue requests from other merchants"""

    def __init__(self, config, model: BaseChatModel):
        super().__init__(MessageTemplate.request())
        self.config = config
        self.model = model
        self.parser = PydanticOutputParser(pydantic_object=Act)
        self.conversation_history = []
        self.merchant_prompt = DIALOGUE_RESPONDER_PROMPT

    def _update_history(self, role: str, content: str):
        """Update conversation history with new message"""
        self.conversation_history.append(f"{role}: {content}")
        if len(self.conversation_history) > 5:
            self.conversation_history.pop(0)

    async def process_interaction(self, interaction_data: dict) -> Act:
        """Process incoming interaction and generate a response"""
        chain = self.merchant_prompt | self.model
        answer = await chain.ainvoke({
            "conversation_history": "\n".join(self.conversation_history),
            "shop_sku": self.agent.shop_sku,
            "current_interaction_type": interaction_data["type"],
            "current_interaction_data": str(interaction_data["data"]),
            "format_instructions": self.parser.get_format_instructions(),
        })
        return self.parser.parse(answer.content)

    async def step(self):
        """Handle incoming dialogue request"""

        def clean_json(text: str) -> str:
            text = text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.endswith("```"):
                text = text[:-3]
            return text.strip()

        cleaned_content = clean_json(self.message.content)

        # Сначала проверяем структуру JSON
        try:
            import json
            data = json.loads(cleaned_content)

            # Если есть поле 'offer' - это Conversate
            if 'offer' in data:
                current_interaction = Conversate.model_validate(data)
            # Если есть поле 'ingredients' - это ShopList
            elif 'ingredients' in data:
                current_interaction = ShopList.model_validate(data)
            else:
                # Если нет ожидаемых полей - ошибка
                await self.context.reply_with_refuse(self.message).with_content(
                    "Invalid request format: missing required fields")
                return

        except json.JSONDecodeError:
            pass

        self._update_history("Opponent", str(current_interaction))
        act = await self.process_interaction(
            {"type": type(current_interaction).__name__, "data": current_interaction.model_dump()}
        )
        self._update_history("Self", str(act.action))

        if isinstance(act.action, ShopList):
            await self.context.reply_with_acknowledge(self.message).with_content(act.action.model_dump_json())
        elif isinstance(act.action, Conversate):
            await self.context.reply_with_acknowledge(self.message).with_content(act.action.model_dump_json())
        else:
            await self.context.reply_with_refuse(self.message).with_content("Unexpected action type")


class AuctionBidderBehaviour(MessageHandlingBehavior):
    """Behavior for handling auction bidding"""

    def __init__(self, config, model: BaseChatModel):
        super().__init__(MessageTemplate.request_proposal())
        self.config = config
        self.model = model
        self.decision_parser = PydanticOutputParser(pydantic_object=Decision)
        self.shoplist_parser = PydanticOutputParser(pydantic_object=ShopList)
        self.merchant_prompt = AUCTION_BIDDER_PROMPT

    async def get_current_board(self) -> ProposalBoard:
        """Retrieve the current state of the proposal board"""
        await asyncio.sleep(self.config.bid_delay)
        await self.context.acknowledge('proposal_board').with_content('')
        response = await self.receive(
            template=MessageTemplate(thread_id=self.context.thread_id, performative=consts.INFORM),
            timeout=60,
        )
        return ProposalBoard.model_validate_json(response.content)

    async def update_my_bid(self):
        """Update the agent's current bid based on the board state"""
        current_board = await self.get_current_board()
        agents_able = ["second_merchant"] if self.context.agent_type == 'first_merchant' else ["first_merchant"]
        chain = self.merchant_prompt | self.model
        await asyncio.sleep(self.config.bid_delay)
        answer = await chain.ainvoke({
            "my_sku": self.agent.shop_sku,
            "current_board": current_board,
            "format_instructions": self.decision_parser.get_format_instructions(),
            "agents_able_to_conversate": agents_able
        })
        try:
            response = self.decision_parser.parse(answer.content).decision
        except Exception as e:
            logger.error("Error parsing decision: %s", e)
            response = ShopList(ingredients={})

        if isinstance(response, ShopList):
            self.agent.current_bid = response
        elif isinstance(response, CollaborationProposal):
            contragent = response.target_agent
            needed = response.needed_ingredients
            start_conversation = StartDialogueBehaviour(
                self.context, self.config, contragent, needed, self.model
            )
            self.agent.add_behaviour(start_conversation)
            await start_conversation.join()
            if self.agent.current_bid is None:
                chain = self.merchant_prompt | self.model
                await asyncio.sleep(self.config.bid_delay)
                fallback_answer = await chain.ainvoke({
                    "my_sku": self.agent.shop_sku,
                    "current_board": current_board,
                    "format_instructions": self.shoplist_parser.get_format_instructions(),
                    "agents_able_to_conversate": agents_able
                })
                self.agent.current_bid = self.shoplist_parser.parse(fallback_answer.content)

    async def get_my_bid(self):
        """Generate and return the agent's current bid"""
        await self.update_my_bid()
        return self.agent.current_bid.model_dump_json()

    async def step(self) -> None:
        """Handle the bidding process"""
        my_bid = await self.get_my_bid()
        await asyncio.sleep(self.config.bid_delay)
        await self.context.reply_with_propose(self.message).with_content(my_bid)
        response = await self.receive(MessageTemplate(thread_id=self.context.thread_id), timeout=15)
        if response.content == 'UPDATING':
            logger.info("Received UPDATING response")
        elif response.performative == consts.ACCEPT:
            pass
        elif response.performative == consts.REFUSE:
            await self.update_my_bid()
        else:
            pass
