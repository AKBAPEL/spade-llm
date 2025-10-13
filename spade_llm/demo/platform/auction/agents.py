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
    agents: List[str] = Field(default_factory=list, description="Names of the agents who proposed")
    proposal: ShopList = Field(description="Current best proposal in ShopList format")
    sku_request: ShopListRequest = Field(description="User sku`s request")
    round_number: int = Field(default=0, description="Текущий номер раунда аукциона")
    total_rounds: int = Field(default=0, description="Всего раундов аукциона")


class ProposalBoardAgentConf(BaseModel):
    total_rounds: int = Field(default=3, description="Общее количество раундов аукциона")
    stable_limit: int = Field(default=3, description="Сколько раундов подряд должна держаться ставка для завершения")


@configuration(ProposalBoardAgentConf)
class ProposalBoardAgent(Agent, Configurable[ProposalBoardAgentConf]):
    """Agent managing the auction proposal board"""
    proposal_board = ProposalBoard(agents=[], proposal=ShopList(), sku_request=ShopListRequest())

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
            self.agent.proposal_board = ProposalBoard(agents=[],
                                                      proposal=ShopList(),
                                                      sku_request=ShopListRequest(ingredients=self.ingredients))
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

            # self.set_is_done()

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
    authors: List[str] = Field(description="Names of the agents who proposed", default_factory=list)
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

    async def get_proposals(self, agents: List[AgentDescription]) -> List[tuple[str, AuctionProposal]]:
        """Collect proposals from agents"""
        sent: Set[str] = set()
        received: Set[str] = set()
        result: List[tuple[str, AuctionProposal]] = []

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
                received.add(response.sender.agent_type)
                prop = AuctionProposal.model_validate_json(response.content)
                result.append((response.sender.agent_type, prop))
        return result

    async def extract_winner_and_notify_losers(self, proposals: List[tuple[str, AuctionProposal]]) -> Optional[
        ShopList]:
        """Select the winning proposal and notify losers"""
        if not proposals:
            return None

        winner_sender = None
        winner = None
        for sender, proposal in proposals:
            if winner is None:
                current_supply = set(self.agent.proposal_board.proposal.ingredients.keys()).intersection(
                    set(self.agent.proposal_board.sku_request.ingredients)
                )
                proposed_supply = set(proposal.prop.ingredients.keys()).intersection(
                    set(self.agent.proposal_board.sku_request.ingredients)
                )

                if not current_supply.issubset(proposed_supply):
                    logger.info("Missing ingredients from current best. Refusing")
                    await self.context.refuse(sender).with_content('')
                elif len(proposed_supply) > len(current_supply) or (
                        len(proposed_supply) == len(current_supply) and
                        sum(proposal.prop.ingredients.values()) < sum(
                    self.agent.proposal_board.proposal.ingredients.values())
                ):
                    logger.info("Better coverage or lower price. Accepting")
                    winner = proposal
                    winner_sender = sender
                    self.agent.proposal_board.proposal = winner.prop
                    self.agent.proposal_board.agents = winner.authors
                    await self.context.accept(sender).with_content('')
                else:
                    logger.info("Not better, rejecting")
                    await self.context.refuse(sender).with_content('')
            else:
                logger.info("Already updated board. Refusing")
                await self.context.refuse(sender).with_content('UPDATING')
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

    def __init__(self, context: AgentContext, config, contragent: str,
                 needed_ingredients: ShopList, model: BaseChatModel):
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

    # -----------------------
    # JSON УТИЛИТЫ
    # -----------------------
    def clean_json(self, text: str) -> str:
        """Удаляет markdown-блоки и возвращает чистый JSON"""
        text = text.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start:end + 1]
        return text.strip()

    def safe_json_load(self, text: str) -> dict:
        """Безопасная загрузка JSON с несколькими попытками восстановления"""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            trimmed = text.split("}")[:-1]
            for i in range(len(trimmed), 0, -1):
                candidate = "}".join(trimmed[:i]) + "}"
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    continue
        return {}

    def fix_act_json(self, data: dict) -> dict:
        """Исправляет структуру JSON от LLM, чтобы гарантировать Act(action={...})"""
        if not isinstance(data, dict):
            return {"action": {"offer": str(data)}}

        action = data.get("action")

        # если LLM не добавил action
        if action is None:
            if "offer" in data:
                return {"action": {"offer": data["offer"]}}
            if "ingredients" in data:
                return {"action": {"ingredients": data["ingredients"]}}
            return {"action": {"offer": str(data)}}

        # если action — строка (Conversate)
        if isinstance(action, str):
            return {"action": {"offer": action}}

        # если словарь с ингредиентами
        if isinstance(action, dict):
            if "ingredients" in action:
                return {"action": {"ingredients": action["ingredients"]}}
            if "offer" in action:
                return {"action": {"offer": action["offer"]}}
            if all(isinstance(v, (int, float)) for v in action.values()):
                return {"action": {"ingredients": action}}
            if all(isinstance(v, str) for v in action.values()):
                return {"action": {"offer": " ".join(action.values())}}

        # fallback
        return {"action": {"offer": str(action)}}

    # -----------------------
    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    # -----------------------
    def _update_history(self, role: str, content: Union[ShopList, Conversate, Act]):
        """Добавляет запись в историю диалога"""
        msg = content.model_dump_json() if hasattr(content, "model_dump_json") else str(content)
        self.conversation_history.append(f"{role}: {msg}")
        if len(self.conversation_history) > 10:
            self.conversation_history.pop(0)

    def _save_dialogue(self):
        """Сохраняет историю диалога в файл"""
        import os, json
        from datetime import datetime
        os.makedirs("dialogues", exist_ok=True)
        timestamp = datetime.now().strftime("%d-%H-%M-%S-%f")
        filename = f"dialogue_start_{timestamp}.txt"
        path = os.path.join("dialogues", filename)

        def parse_saved(content: str):
            try:
                data = json.loads(content)
                if "offer" in data:
                    return "Conversate", data["offer"]
                if "ingredients" in data:
                    return "ShopList", str(data["ingredients"])
            except json.JSONDecodeError:
                pass
            return "Unknown", content

        with open(path, "w", encoding="utf-8") as f:
            f.write(f"=== Dialogue log started at {timestamp} ===\n")
            f.write(f"Initiator agent: {self.context.agent_type}\n")
            f.write(f"Contragent agent: {self.contragent}\n")
            f.write("=========================================\n\n")
            for entry in self.conversation_history:
                try:
                    role, content = entry.split(": ", 1)
                    msg_type, display = parse_saved(content)
                    # определяем тип агента
                    if role.lower() == "self":
                        agent_type = self.context.agent_type
                    else:
                        agent_type = self.contragent

                    f.write(f"{role} [{agent_type}] ({msg_type}): {display}\n")
                except Exception:
                    f.write(f"ParseError: {entry}\n")

        logger.info("Saved dialogue to %s", path)

    # -----------------------
    # ГЛАВНАЯ ЛОГИКА
    # -----------------------
    async def process_response(self, interaction_data: dict) -> Act:
        """Обрабатывает ответ контрагента через модель"""
        chain = self.merchant_prompt | self.model
        answer = await chain.ainvoke({
            "conversation_history": "\n".join(self.conversation_history),
            "shop_sku": self.agent.shop_sku,
            "current_interaction_type": interaction_data["type"],
            "current_interaction_data": str(interaction_data["data"]),
            "format_instructions": self.parser.get_format_instructions(),
        })

        # print("==== RAW LLM RESPONSE INITIATOR ====")
        # print(answer.content)

        cleaned = self.clean_json(answer.content)
        data = self.safe_json_load(cleaned)
        fixed = self.fix_act_json(data)

        try:
            return self.parser.parse(json.dumps(fixed))
        except Exception as e:
            logger.error("Error parsing act: %s\nRecovered JSON: %s", e, fixed)
            return Act(action=Conversate(offer="Извините, я не смог корректно ответить."))

    async def step(self):
        """Основной процесс инициации диалога"""
        logger.info("Starting dialogue with %s", self.contragent)
        thread = await self.context.fork_thread()
        max_iterations = 10
        iteration = 0

        # начинаем с исходного ShopList-запроса
        self._update_history("Self", self.needed_ingredients)
        current_offer = self.needed_ingredients

        while iteration < max_iterations:
            # отправка текущего предложения
            if isinstance(current_offer, (ShopList, Conversate)):
                await thread.request(self.contragent).with_content(current_offer.model_dump_json())
            else:
                logger.warning("Unexpected offer type: %s", type(current_offer))
                break

            # ожидание ответа
            response = await self.receive(
                template=MessageTemplate(thread_id=thread.thread_id,
                                         performative=consts.ACKNOWLEDGE),
                timeout=60
            )
            if not response:
                logger.warning("No response from %s", self.contragent)
                break

            # попытка распарсить ответ
            cleaned = self.clean_json(response.content)
            data = self.safe_json_load(cleaned)

            if not data:
                logger.error("Empty or invalid response JSON from %s: %s", self.contragent, response.content)
                break

            competitor_msg = None
            if "ingredients" in data:
                competitor_msg = ShopList.model_validate(data)
                msg_type = "ShopList"
            elif "offer" in data:
                competitor_msg = Conversate.model_validate(data)
                msg_type = "Conversate"
            elif "action" in data:
                act_data = data["action"]
                if "ingredients" in act_data:
                    competitor_msg = ShopList.model_validate(act_data)
                    msg_type = "ShopList"
                elif "offer" in act_data:
                    competitor_msg = Conversate.model_validate(act_data)
                    msg_type = "Conversate"

            if competitor_msg is None:
                logger.error("Unrecognized message from %s: %s", self.contragent, data)
                break

            # print("++++ INCOMING RESPONSE INITIATOR ++++\n", competitor_msg)
            self._update_history("Opponent", competitor_msg)
            # print('===== COMPETITOR MESSAGE\n', competitor_msg)
            # если получили ShopList — контрагент согласен
            if isinstance(competitor_msg, ShopList):
                combined = dict(self.agent.shop_sku)
                overlaps = set(combined.keys()) & set(competitor_msg.ingredients.keys())
                for k in overlaps:
                    combined[k] = min(combined[k], competitor_msg.ingredients[k])
                combined.update({k: v for k, v in competitor_msg.ingredients.items()
                                 if k not in self.agent.shop_sku})

                self.agent.collaborators = sorted([self.context.agent_type, self.contragent])
                self.agent.current_bid = AuctionProposal(
                    authors=self.agent.collaborators,
                    prop=ShopList(ingredients=combined)
                )
                break

            # если получили Conversate — продолжаем торг
            elif isinstance(competitor_msg, Conversate):
                act = await self.process_response({
                    "type": msg_type,
                    "data": competitor_msg.model_dump(),
                })
                self._update_history("Self", act.action)
                current_offer = act.action
                # print('MY MESSAGE   ', current_offer)
                if isinstance(act.action, ShopList):
                    await thread.acknowledge(self.contragent).with_content(act.action.model_dump_json())
                elif isinstance(act.action, Conversate):
                    await thread.acknowledge(self.contragent).with_content(act.action.model_dump_json())
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
        self.conversation_history = dict()
        self.merchant_prompt = DIALOGUE_RESPONDER_PROMPT

    # -----------------------
    # JSON УТИЛИТЫ
    # -----------------------
    def clean_json(self, text: str) -> str:
        """Удаляет markdown-блоки и лишние символы."""
        text = text.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        # Иногда LLM вставляет лишние запятые или \n перед JSON
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx != -1:
            text = text[start_idx:end_idx + 1]
        return text.strip()

    def safe_json_load(self, text: str) -> dict:
        """Пытается корректно загрузить JSON с несколькими попытками исправления."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Попробуем подрезать строку до последней закрывающей фигурной скобки
            trimmed = text.split("}")[:-1]
            for i in range(len(trimmed), 0, -1):
                try_text = "}".join(trimmed[:i]) + "}"
                try:
                    return json.loads(try_text)
                except json.JSONDecodeError:
                    continue
        return {}

    def fix_act_json(self, data: dict) -> dict:
        """Исправляет структуру Act JSON при некорректных вложениях."""
        if not isinstance(data, dict):
            return {"action": {"offer": str(data)}}

        action = data.get("action")

        # Если LLM не добавил ключ action, пытаемся распознать вручную
        if action is None:
            if "offer" in data:
                data = {"action": {"offer": data["offer"]}}
            elif "ingredients" in data:
                data = {"action": {"ingredients": data["ingredients"]}}
            else:
                data = {"action": {"offer": str(data)}}
            return data

        # Если action строка — это Conversate
        if isinstance(action, str):
            return {"action": {"offer": action}}

        # Если action — словарь ингредиентов
        if isinstance(action, dict):
            if "ingredients" in action:
                return {"action": {"ingredients": action["ingredients"]}}
            elif "offer" in action:
                return {"action": {"offer": action["offer"]}}
            elif all(isinstance(v, (int, float)) for v in action.values()):
                return {"action": {"ingredients": action}}
            elif all(isinstance(v, str) for v in action.values()):
                return {"action": {"offer": " ".join(action.values())}}

        # fallback
        return {"action": {"offer": str(action)}}

    # -----------------------
    # ИСТОРИЯ
    # -----------------------
    def _update_history(self, role: str, content: str, conversation_id):
        """Добавляет в историю реплику."""
        self.conversation_history[conversation_id].append(f"{role}: {content}")
        if len(self.conversation_history[conversation_id]) > 10:
            self.conversation_history[conversation_id].pop(0)

    # -----------------------
    # ГЛАВНАЯ ЛОГИКА
    # -----------------------
    async def process_interaction(self, interaction_data: dict, conversation_id) -> Act:
        """Обрабатывает запрос и получает ответ от модели."""
        chain = self.merchant_prompt | self.model
        answer = await chain.ainvoke({
            "conversation_history": "\n".join(self.conversation_history[conversation_id]),
            "shop_sku": self.agent.shop_sku,
            "current_interaction_type": interaction_data["type"],
            "current_interaction_data": str(interaction_data["data"]),
            "format_instructions": self.parser.get_format_instructions(),
        })

        # print("==== RAW LLM RESPONSE ====")
        # print(answer.content)

        cleaned = self.clean_json(answer.content)
        data = self.safe_json_load(cleaned)
        fixed = self.fix_act_json(data)
        try:
            return self.parser.parse(json.dumps(fixed))
        except Exception as e:
            logger.error("Error parsing Act: %s\nRecovered JSON: %s", e, fixed)
            return Act(action=Conversate(offer="Извините, я не смог корректно ответить."))

    async def step(self):
        """Обработка входящего сообщения диалога."""
        raw_content = self.message.content.strip()
        conversation_id = self.message.thread_id
        if conversation_id not in self.conversation_history.keys():
            self.conversation_history[conversation_id] = list()

        # print('\n\n\n  ++  initiator is', self.message.sender, '\n\n\n------------',
        #       len(self.conversation_history[conversation_id]), '------',
        #       self.message.thread_id)

        cleaned = self.clean_json(raw_content)
        # Разбор входящего JSON
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            await self.context.reply_with_refuse(self.message).with_content("Некорректный формат JSON")
            return

        # Определяем тип взаимодействия
        current_interaction = None
        if "offer" in data:
            current_interaction = Conversate.model_validate(data)
        elif "ingredients" in data:
            current_interaction = ShopList.model_validate(data)
        elif "action" in data:
            # LLM мог обернуть действием
            action_data = data["action"]
            if "offer" in action_data:
                current_interaction = Conversate.model_validate(action_data)
            elif "ingredients" in action_data:
                current_interaction = ShopList.model_validate(action_data)
        else:
            await self.context.reply_with_refuse(self.message).with_content(
                "Invalid request format: missing required fields")
            return

        # print("++++ INCOMING INTERACTION ++++\n", current_interaction)

        self._update_history("Opponent", str(current_interaction), conversation_id)
        act = await self.process_interaction(
            {
                "type": type(current_interaction).__name__,
                "data": current_interaction.model_dump(),
            },
            conversation_id)
        self._update_history("Self", str(act.action), conversation_id)
        # print("---- RESPONSE ACT ----\n", act.action)

        # Отправка корректного ответа
        if isinstance(act.action, ShopList):
            await self.context.reply_with_acknowledge(self.message).with_content(
                act.action.model_dump_json()
            )
        elif isinstance(act.action, Conversate):
            await self.context.reply_with_acknowledge(self.message).with_content(
                act.action.model_dump_json()
            )
        else:
            await self.context.reply_with_refuse(self.message).with_content(
                "Unexpected action type"
            )


class AuctionBidderBehaviour(MessageHandlingBehavior):
    """Behavior for handling auction bidding"""

    def __init__(self, config, model: BaseChatModel):
        super().__init__(MessageTemplate.request_proposal())
        self.config = config
        self.model = model
        self.decision_parser = PydanticOutputParser(pydantic_object=Decision)
        self.shoplist_parser = PydanticOutputParser(pydantic_object=ShopList)
        self.merchant_prompt = AUCTION_BIDDER_PROMPT

    def clean_json(self, text: str) -> str:
        """Clean the JSON string by removing markdown code blocks and extra text."""
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[: -3]
        return text.strip()

    def safe_json_loads(self, text: str) -> Optional[dict]:
        """
        Безопасно парсит JSON-ответ от LLM.
        Поддерживает частично повреждённые или вложенные структуры.
        Возвращает dict или None.
        """
        text = self.clean_json(text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            import re
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except Exception:
                    pass
            logger.error("safe_json_loads(): не удалось распарсить JSON\n%s", text)
            return None

    def normalize_llm_output(self, data: Optional[dict]) -> dict:
        """
        Приводит любые варианты ответов LLM к единому виду:
        - Если это сразу ShopList — оборачиваем.
        - Если это CollaborationProposal — оставляем.
        - Если LLM вложил структуру в лишние уровни ("decision", "CollaborationProposal" и т.п.) — разворачиваем.
        """
        if not data:
            return {"decision": {"ingredients": {}}}

        if "decision" in data:
            dec = data["decision"]
        else:
            dec = data

        if isinstance(dec, dict) and "ingredients" in dec:
            return {"decision": dec}

        if isinstance(dec, dict) and "target_agent" in dec and "needed_ingredients" in dec:
            needed = dec["needed_ingredients"]
            if isinstance(needed, dict) and "ingredients" not in needed:
                needed = {"ingredients": needed}
            return {"decision": {"target_agent": dec["target_agent"], "needed_ingredients": needed}}

        for k, v in dec.items() if isinstance(dec, dict) else []:
            if isinstance(v, dict):
                if "ingredients" in v or ("target_agent" in v and "needed_ingredients" in v):
                    return {"decision": v}

        if all(isinstance(k, str) for k in dec.keys()) and all(isinstance(v, (int, float)) for v in dec.values()):
            return {"decision": {"ingredients": dec}}

        return {"decision": {"ingredients": {}}}

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
        if self.context.agent_type not in current_board.agents:
            agents_able = [a for a in ["first_merchant", "second_merchant"] if a != self.context.agent_type]
            chain = self.merchant_prompt | self.model
            await asyncio.sleep(self.config.bid_delay)
            answer = await chain.ainvoke({
                "my_sku": self.agent.shop_sku,
                "current_board": current_board,
                "format_instructions": self.decision_parser.get_format_instructions(),
                "agents_able_to_conversate": agents_able
            })

            # print('000000 НАЧАЛО \n', answer.content)
            try:
                raw_data = self.safe_json_loads(answer.content)
                normalized = self.normalize_llm_output(raw_data)
                response = self.decision_parser.parse(json.dumps(normalized)).decision
            except Exception as e:
                logger.error("Ошибка при обработке ответа LLM: %s", e)
                response = ShopList(ingredients={})

            if isinstance(response, ShopList):
                self.agent.current_bid = AuctionProposal(
                    authors=[self.context.agent_type],
                    prop=response
                )
            elif isinstance(response, CollaborationProposal):
                contragent = response.target_agent
                print("DIALOGUE STARTED WITH CONTRAGENT", contragent)
                needed = response.needed_ingredients
                start_conversation = StartDialogueBehaviour(
                    self.context, self.config, contragent, needed, self.model
                )
                self.agent.add_behaviour(start_conversation)
                await start_conversation.join()
                print("DIALOGUE FINISHED WITH CONTRAGENT", contragent)
                if self.agent.current_bid is None:
                    chain = self.merchant_prompt | self.model
                    await asyncio.sleep(self.config.bid_delay)
                    fallback_answer = await chain.ainvoke({
                        "my_sku": self.agent.shop_sku,
                        "current_board": current_board,
                        "format_instructions": self.shoplist_parser.get_format_instructions(),
                        "agents_able_to_conversate": agents_able,
                        "self_type": self.context.agent_type
                    })
                    fallback_data = self.safe_json_loads(fallback_answer.content)
                    normalized_fallback = self.normalize_llm_output(fallback_data)
                    fallback_prop = self.shoplist_parser.parse(json.dumps(normalized_fallback))
                    self.agent.current_bid = AuctionProposal(
                        authors=[self.context.agent_type],
                        prop=fallback_prop
                    )

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
