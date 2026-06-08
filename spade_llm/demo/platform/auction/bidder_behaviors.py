import asyncio
import json
import logging
import re
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser

from spade_llm import consts
from spade_llm.core.behaviors import MessageHandlingBehavior, MessageTemplate
from spade_llm.demo.platform.auction.agent_prompts import AUCTION_BIDDER_PROMPT
from spade_llm.demo.platform.auction.models import (
    AuctionProposal,
    CollaborationProposal,
    Decision,
    ProposalBoard,
    ShopList,
)

logger = logging.getLogger(__name__)


class AuctionBidderBehaviour(MessageHandlingBehavior):
    """Behavior for handling auction bidding"""

    def __init__(self, config, model: BaseChatModel):
        super().__init__(MessageTemplate.request_proposal())
        self.config = config
        self.model = model
        self.decision_parser = PydanticOutputParser(pydantic_object=Decision)
        self.shoplist_parser = PydanticOutputParser(pydantic_object=ShopList)
        self.bidder_prompt = AUCTION_BIDDER_PROMPT

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
            template=MessageTemplate(thread_id=self.context.thread_id, performative=consts.INFORM), timeout=15)
        if not response:
            logger.error("No info about auction board after 15 seconds")
            return None
        return ProposalBoard.model_validate_json(response.content)

    def _build_trust_context(self) -> str:
        """Build trust context string from agent's trust memory for available partners."""
        if not getattr(self.agent.config, 'enable_trust_mechanism', False):
            return ""
        lines = ["Твои прошлые впечатления о партнёрах (от новых к старым):"]
        has_any = False
        for partner in ["first_merchant", "second_merchant", "third_merchant"]:
            if partner == self.context.agent_type:
                continue
            records = self.agent.trust_memory.get(partner, [])
            if records:
                latest = records[-1]
                lines.append(f"- {partner} ({latest.timestamp}): {latest.impression}")
                has_any = True
            else:
                lines.append(f"- {partner}: Прошлого опыта взаимодействия нет. Это первая встреча.")
        if not has_any:
            return "Прошлого опыта взаимодействия ни с одним агентом нет. Все партнёры равнозначны."
        return "\n".join(lines)

    async def update_my_bid(self):
        """Update the agent's current bid based on the board state"""
        current_board = await self.get_current_board()
        if self.context.agent_type not in current_board.agents:
            agents_able = [a for a in ["first_merchant", "second_merchant"] if a != self.context.agent_type]
            trust_context = self._build_trust_context()
            chain = self.bidder_prompt | self.model
            await asyncio.sleep(self.config.bid_delay)
            answer = await chain.ainvoke({
                "my_sku": self.agent.shop_sku,
                "current_board": current_board,
                "format_instructions": self.decision_parser.get_format_instructions(),
                "agents_able_to_conversate": agents_able,
                "user_max_price": current_board.user_wants_lower_than,
                "trust_context": trust_context,
            })

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
                    prop=response,
                    bid_ingredient_split={self.context.agent_type: response.ingredients}
                )
                print("BID UPDATED Instantly")
            elif isinstance(response, CollaborationProposal):
                contragent = response.target_agent
                print("DIALOGUE STARTED WITH CONTRAGENT", contragent)
                needed = response.needed_ingredients
                # Импорт здесь, чтобы избежать циклического импорта на уровне модуля
                from spade_llm.demo.platform.auction.dialogue_behaviors import StartDialogueBehaviour
                start_conversation = StartDialogueBehaviour(
                    self.context, self.config, contragent, needed, self.model, current_board.sku_request,
                    current_board.user_wants_lower_than
                )
                self.agent.add_behaviour(start_conversation)
                await start_conversation.join()
                print("DIALOGUE FINISHED WITH CONTRAGENT", contragent)
                if self.agent.current_bid is None:
                    self.agent.current_bid = AuctionProposal(
                        authors=[self.context.agent_type],
                        prop=ShopList(ingredients={}),
                        bid_ingredient_split={self.context.agent_type: {}}
                    )
                    print("BID UPDATED FALLBACK")
        else:
            print("ALREADY IN WINNING POSITION")

    async def get_my_bid(self):
        """Generate and return the agent's current bid"""
        await self.update_my_bid()
        if self.agent.current_bid is None:
            logger.warning("Current bid is None")
            self.agent.current_bid = AuctionProposal(
                authors=[self.context.agent_type],
                prop=ShopList(ingredients={}),
                bid_ingredient_split={self.context.agent_type: {}}
            )
            print("BID UPDATED EMPTY")
        return self.agent.current_bid.model_dump_json()

    async def step(self) -> None:
        """Handle the bidding process"""
        msg = self.message
        if msg:
            my_bid = await self.get_my_bid()
            await asyncio.sleep(self.config.bid_delay)
            await self.context.reply_with_propose(msg).with_content(my_bid)
            response = await self.receive(MessageTemplate(thread_id=self.context.thread_id), timeout=15)
            if response is None:
                logger.warning("No response received from %s to bidder %s", msg.sender, self.context.agent_type)
            elif response.content == 'UPDATING':
                logger.info("Received UPDATING response")
            elif response.performative == consts.ACCEPT:
                pass
            elif response.performative == consts.REFUSE:
                await self.update_my_bid()
            else:
                pass
