import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Union

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from spade_llm import consts
from spade_llm.core.behaviors import ContextBehaviour, MessageHandlingBehavior, MessageTemplate
from spade_llm.demo.platform.auction.agent_prompts import (
    DIALOGUE_INITIATOR_PROMPT,
    DIALOGUE_RESPONDER_PROMPT,
    NEGOTIATION_DECISION_PROMPT,
)
from spade_llm.demo.platform.auction.models import (
    Act,
    AuctionProposal,
    Conversate,
    NegotiationResponse,
    ShopList,
    ShopListRequest,
)

logger = logging.getLogger(__name__)


class StartDialogueBehaviour(ContextBehaviour):
    """Behavior for initiating a dialogue with another merchant"""

    def __init__(self, context, config, contragent: str,
                 needed_ingredients: ShopList, model: BaseChatModel, user_request: ShopListRequest,
                 user_max_price: int):
        super().__init__(context)
        self.config = config
        self.contragent = contragent
        self.needed_ingredients = needed_ingredients
        self.model = model
        self.conversation_history = []
        self.full_conversation_history: List[str] = []
        self.parser = PydanticOutputParser(pydantic_object=Act)
        self.shoplist_parser = PydanticOutputParser(pydantic_object=ShopList)
        self.conversate_parser = PydanticOutputParser(pydantic_object=Conversate)
        self.initiator_prompt = DIALOGUE_INITIATOR_PROMPT
        self.user_request = user_request
        self.user_max_price = user_max_price

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
        if content is None:
            logger.error('NONE 3')
        msg = content.model_dump_json() if hasattr(content, "model_dump_json") else str(content)
        entry = f"{role}: {msg}"
        self.full_conversation_history.append(entry)
        self.conversation_history.append(entry)
        if len(self.conversation_history) > 10:
            self.conversation_history.pop(0)

    def _save_dialogue(self):
        """Сохраняет историю диалога в файл"""
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
            for entry in self.full_conversation_history:
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
        chain = self.initiator_prompt | self.model
        answer = await chain.ainvoke({
            "conversation_history": "\n".join(self.conversation_history),
            "shop_sku": self.agent.shop_sku,
            "current_interaction_type": interaction_data["type"],
            "current_interaction_data": str(interaction_data["data"]),
            "format_instructions": self.parser.get_format_instructions(),
            "user_max_price": self.user_max_price,
            "user_request": self.user_request.model_dump_json()  # Для LLM
        })

        cleaned = self.clean_json(answer.content)
        data = self.safe_json_load(cleaned)
        fixed = self.fix_act_json(data)

        try:
            return self.parser.parse(json.dumps(fixed))
        except Exception as e:
            logger.error("Error parsing act: %s\nRecovered JSON: %s", e, fixed)
            return Act(action=Conversate(offer="Извините, я не смог корректно ответить."))

    async def generate_final_prices(self, dialogue_initiator_ingredients: Dict[str, int],
                                    dialogue_contragent_ingredients: Dict[str, int]) -> ShopList:
        """
        Определяет цены инициатора для своей части ингредиентов с помощью LLM,
        используя контекст прошедшего диалога.
        """
        prompt = ChatPromptTemplate.from_template(
            """Ты — агент магазина, завершивший переговоры с контрагентом и готовящий итоговую часть ставки.

            На основе истории переговоров и согласованных ингредиентов рассчитай цены для своей доли в совместной ставке.

            История переговоров:
            {conversation_history}

            Твой ассортимент и базовые цены:
            {shop_sku}

            Ингредиенты, которые поставляешь ты:
            {dialogue_initiator_ingredients}

            Ингредиенты которые поставляет контрагент:
            {dialogue_contragent_ingredients}


            Условия:
            - Твоя цель — получить прибыль, но при этом сохранить конкурентоспособность ставки.
            - Можешь сделать наценку относительно базовой цены:
                • 5–10% — если контрагент сильно давил на снижение цен или соглашение достигалось трудно.  
                • 10–20% — если переговоры прошли в твою пользу или контрагент согласился быстро.
            - Не ставь слишком высокую наценку (>25%), иначе:
                • пользователь может не принять ставку,
                • или ваша совместная ставка проиграет в аукционе.
            - Рассчитывай реалистичные цены, чтобы предложение выглядело разумным и имело шанс выиграть.
            - Возвращай только итоговые цены ТОЛЬКО для своей части ингредиентов.

             Дополнительное условие:
            Если переменная {user_max_price} не равна None, это означает, что пользователь отказался покупать набор по прежней цене.
            Он готов заплатить максимум сумму {user_max_price}.
            Поэтому:
            - Итоговая общая стоимость (твои ингредиенты + ингредиенты контрагента) должна быть строго меньше этого лимита.
            - Если ваша общая цена превысит {user_max_price}, пользователь просто не купит набор.
            - В этом случае ставка будет отклонена, и ни ты, ни контрагент не получите прибыль.
            - Рассчитай цены так, чтобы совместная ставка выглядела выгодно, уложилась в лимит пользователя и при этом приносила тебе умеренную прибыль (5–15%).
            - Если общая сумма превысит лимит {user_max_price}, ставка будет автоматически отклонена системой и сделка не состоится.
            """
        )

        class ShopListItem(BaseModel):
            ingredient: str = Field(description="название ингредиента")
            price: int = Field(description="цена за этот ингредиент")

        class ShopListResponse(BaseModel):
            """ Список ингредиентов и их цен """
            ingredients: List[ShopListItem] = Field(
                description="Список товаров с ценами")  # Вспомогательные структуры. Поле не может быть Dict для structured output

        structured_llm = prompt | self.model.with_structured_output(ShopListResponse)
        answer = await structured_llm.ainvoke({
            "conversation_history": "\n".join(self.conversation_history),
            "shop_sku": self.agent.shop_sku,
            "dialogue_initiator_ingredients": dialogue_initiator_ingredients,
            "dialogue_contragent_ingredients": dialogue_contragent_ingredients,
            "user_max_price": self.user_max_price
        })
        result = ShopList(ingredients={item.ingredient: item.price for item in answer.ingredients})
        return result

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
                if current_offer is None:
                    logger.error('NONE 4')
                message_data = {
                    "action": current_offer.model_dump(),
                    "user_request": self.user_request.model_dump()  # Вот и всё!
                }
                await thread.request(self.contragent).with_content(json.dumps(message_data))
            else:
                logger.warning("Unexpected offer type: %s", type(current_offer))
                break

            # ожидание ответа (ACKNOWLEDGE или REFUSE)
            response = await self.receive(
                template=MessageTemplate(thread_id=thread.thread_id),
                timeout=60
            )
            if not response:
                logger.warning("No response from %s", self.contragent)
                break

            if response.performative == consts.REFUSE:
                logger.info("Contragent %s refused negotiation. Reason: %s", self.contragent, response.content)
                break

            if response.performative != consts.ACKNOWLEDGE:
                logger.warning("Unexpected performative from %s: %s", self.contragent, response.performative)
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

            self._update_history("Opponent", competitor_msg)
            if isinstance(competitor_msg, ShopList):
                # Логику множеств писал я :-)
                A = set(self.user_request.ingredients)  # множество ингредиентов, которые нужны пользователю
                B = self.agent.shop_sku  # словарь ассортимента инициатора {ингредиент: цена}
                C = competitor_msg.ingredients  # словарь ингредиентов, предложенных контрагентом

                # С & A - ингредиенты от контрагента, которые идут в заказ
                contragent_in_order = {item: price for item, price in C.items() if item in A}

                # С \ (С & A) - ингредиенты, которые контрагент дал сверх того что нужны пользователю
                extras = {item: price for item, price in C.items() if item not in A}
                if extras:
                    logger.info(f"Agent {self.contragent} sold extras: {extras} to {self.context.agent_type}")
                # B & (A \ (С & A)) - ингредиенты, которые поставляет инициатор диалога
                initiator_in_order = {item: price for item, price in B.items()
                                      if item in A and item not in contragent_in_order}

                # (B & (A \ (С & A)))^* - ингредиенты, которые поставляет инициатор диалога с обновленными ценами
                initiator_final_prices = await self.generate_final_prices(initiator_in_order, contragent_in_order)

                # Совместная ставка
                combined = {**initiator_final_prices.ingredients, **contragent_in_order}

                # Обновляем ставку агента
                self.agent.current_bid = AuctionProposal(
                    authors=sorted([self.context.agent_type, self.contragent]),
                    prop=ShopList(ingredients=combined),
                    bid_ingredient_split={
                        self.context.agent_type: initiator_final_prices.ingredients,
                        self.contragent: contragent_in_order
                    }
                )
                print("BID UPDATED COLLABORATION")
                break

            # если получили Conversate — продолжаем торг
            elif isinstance(competitor_msg, Conversate):
                if competitor_msg is None:
                    logger.error('NONE 6')
                act = await self.process_response({
                    "type": msg_type,
                    "data": competitor_msg.model_dump(),
                })
                self._update_history("Self", act.action)
                current_offer = act.action
                if act.action is None:
                    logger.error('NONE 7')
                if isinstance(act.action, ShopList) or isinstance(act.action, Conversate):
                    response_data = {
                        "action": act.action.model_dump(),
                        "user_request": self.user_request.model_dump()
                    }
                    await thread.acknowledge(self.contragent).with_content(json.dumps(response_data))
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
        self.full_conversation_history: Dict[str, List[str]] = {}
        self.merchant_prompt = DIALOGUE_RESPONDER_PROMPT
        self.user_requests: Dict[str, ShopListRequest] = {}  # conversation_id -> запрос
        self._negotiation_decisions: Dict[str, NegotiationResponse] = {}  # conversation_id -> решение о переговорах

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
        entry = f"{role}: {content}"
        if conversation_id not in self.full_conversation_history:
            self.full_conversation_history[conversation_id] = []
        self.full_conversation_history[conversation_id].append(entry)
        self.conversation_history[conversation_id].append(entry)
        if len(self.conversation_history[conversation_id]) > 10:
            self.conversation_history[conversation_id].pop(0)

    # -----------------------
    # ФИЧА: ОТКАЗ ОТ ПЕРЕГОВОРОВ
    # -----------------------
    async def _should_negotiate(self, sender_id: str) -> NegotiationResponse:
        """Спрашивает LLM, хочет ли агент вступить в переговоры с отправителем."""
        parser = PydanticOutputParser(pydantic_object=NegotiationResponse)
        chain = NEGOTIATION_DECISION_PROMPT | self.model
        answer = await chain.ainvoke({
            "sender_id": sender_id,
            "shop_sku": self.agent.shop_sku,
            "format_instructions": parser.get_format_instructions()
        })
        cleaned = self.clean_json(answer.content)
        data = self.safe_json_load(cleaned)
        try:
            return NegotiationResponse.model_validate(data)
        except Exception as e:
            logger.error("Error parsing negotiation response: %s", e)
            return NegotiationResponse(agree=True, reason="Ошибка парсинга, соглашаемся по умолчанию")

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
            "user_request": self.user_requests[conversation_id].model_dump_json()
        })

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
        msg = self.message
        raw_content = msg.content.strip()
        conversation_id = msg.thread_id

        # Проверка — хочет ли агент вступить в переговоры?
        # Выполняется только при первом сообщении в рамках конкретного thread_id
        if conversation_id not in self._negotiation_decisions:
            negotiation_decision = await self._should_negotiate(msg.sender.agent_type)
            self._negotiation_decisions[conversation_id] = negotiation_decision
            if not negotiation_decision.agree:
                logger.info(
                    "Agent %s refused negotiation with %s: %s",
                    self.context.agent_type, msg.sender.agent_type, negotiation_decision.reason
                )
                await self.context.reply_with_refuse(msg).with_content(negotiation_decision.reason)
                return
            else:
                logger.info(
                    "Agent %s agreed negotiation with %s: %s",
                    self.context.agent_type, msg.sender.agent_type, negotiation_decision.reason
                )

        if conversation_id not in self.conversation_history.keys():
            self.conversation_history[conversation_id] = list()

        cleaned = self.clean_json(raw_content)
        # Разбор входящего JSON
        try:
            data = json.loads(cleaned)
            user_request_data = data.get("user_request", {})
            if user_request_data:
                self.user_requests[conversation_id] = ShopListRequest.model_validate(user_request_data)
        except json.JSONDecodeError:
            await self.context.reply_with_refuse(msg).with_content("Некорректный формат JSON")

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
            await self.context.reply_with_refuse(msg).with_content(
                "Invalid request format: missing required fields")

        self._update_history("Opponent", str(current_interaction), conversation_id)
        if current_interaction is None:
            logger.error('NONE 9')
        act = await self.process_interaction(
            {
                "type": type(current_interaction).__name__,
                "data": current_interaction.model_dump(),
            },
            conversation_id)
        self._update_history("Self", str(act.action), conversation_id)

        if act.action is None:
            logger.error('NONE 10')
        # Отправка корректного ответа
        if isinstance(act.action, ShopList):
            await self.context.reply_with_acknowledge(msg).with_content(
                act.action.model_dump_json()
            )
        elif isinstance(act.action, Conversate):
            await self.context.reply_with_acknowledge(msg).with_content(
                act.action.model_dump_json()
            )
        else:
            await self.context.reply_with_refuse(msg).with_content(
                "Unexpected action type"
            )
