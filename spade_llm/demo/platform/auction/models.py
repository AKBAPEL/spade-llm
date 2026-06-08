from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field


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


class ProposalBoard(BaseModel):
    agents: List[str] = Field(default_factory=list, description="Names of the agents who proposed")
    proposal: ShopList = Field(description="Current best proposal in ShopList format")
    sku_request: ShopListRequest = Field(description="User sku`s request")
    round_number: int = Field(default=0, description="Текущий номер раунда аукциона")
    total_rounds: int = Field(default=0, description="Всего раундов аукциона")
    bid_ingredient_split: Dict[str, Dict[str, int]] = Field(
        default={},
        description="Указывает на разделение ингредиентов для ставки между агентами"
    )
    user_wants_lower_than: Optional[int] = Field(
        default=None,
        description="Пользователь может заплатить за заказ только меньше, чем это значение "
    )


class ProposalBoardAgentConf(BaseModel):
    total_rounds: int = Field(default=3, description="Общее количество раундов аукциона")
    stable_limit: int = Field(default=3, description="Сколько раундов подряд должна держаться ставка для завершения")
    max_price: int = Field(default=1e9, description="Максимальная цена, которую готов заплатить пользователь")


class AuctionProposal(BaseModel):
    authors: List[str] = Field(description="Names of the agents who proposed", default_factory=list)
    prop: ShopList = Field(description="proposal_shoplist")
    bid_ingredient_split: Dict[str, Dict[str, int]] = Field(
        default={},
        description="Указывает на разделение ингредиентов для ставки между агентами"
    )


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


class NegotiationResponse(BaseModel):
    """Response to a negotiation invitation."""
    agree: bool = Field(description="Согласен ли вступить в переговоры")
    reason: str = Field(description="Причина решения", default="")


class TrustImpressionRecord(BaseModel):
    """Single trust impression record with timestamp and metadata."""
    timestamp: str = Field(description="ISO timestamp of the impression")
    impression: str = Field(description="The impression text")
    outcome: str = Field(description="Outcome of the dialogue: success, refused, round_limit, error, timeout, unknown/preload")
    source: str = Field(description="Source of the impression: runtime, preload")
