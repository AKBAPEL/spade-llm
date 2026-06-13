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
    system_trust_enabled: bool = Field(default=False, description="Enable platform/system trust board")
    trust_reject_threshold: float = Field(default=2.0, description="Average rating below this rejects bids")
    min_reviews_for_reject: int = Field(default=5, description="Minimum reviews before low rating triggers rejection")
    trust_board_persist: bool = Field(default=True, description="Persist trust board to disk")
    trust_board_path: str = Field(default="data/memory/system_trust_board.json", description="Path to persist trust board")


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
                    "Ты либо ПОКУПАТЕЛЬ (инициатор), либо ПРОДАВЕЦ (респондент). "
                    "ShopList используется для финального согласия цен. Conversate — для торга. "
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


class SystemTrustReview(BaseModel):
    """Single review on the platform trust board."""
    reviewer_id: str = Field(description="Agent that left the review")
    target_agent: str = Field(description="Agent being reviewed")
    rating: int = Field(description="Rating from 1 to 5 stars", ge=1, le=5)
    comment: str = Field(description="Short comment up to 30 characters about your conversation", max_length=30)
    outcome: str = Field(description="Outcome of the dialogue: success, refused, round_limit, error, timeout")
    timestamp: Optional[str] = Field(default=None, description="ISO timestamp of the review")


class TrustScoreRequest(BaseModel):
    """Request for a partner's trust score."""
    target_agent: str = Field(description="Agent whose trust score is requested")


class TrustScoreResponse(BaseModel):
    """Trust score and recent reviews for a partner."""
    target_agent: str = Field(description="Agent being reviewed")
    average_rating: float = Field(description="Average rating from 1 to 5")
    total_reviews: int = Field(description="Total number of reviews")
    positive_count: int = Field(description="Number of positive reviews (4-5 stars)")
    negative_count: int = Field(description="Number of negative reviews (1-3 stars)")
    positive_reviews: List[SystemTrustReview] = Field(
        default_factory=list,
        description="Up to 5 most recent positive reviews (4-5 stars)"
    )
    negative_reviews: List[SystemTrustReview] = Field(
        default_factory=list,
        description="Up to 5 most recent negative reviews (1-3 stars)"
    )


class TrustFeedback(BaseModel):
    """Feedback sent by a merchant after a dialogue."""
    reviewer_id: str = Field(description="Agent that leaves the feedback")
    target_agent: str = Field(description="Agent being reviewed")
    rating: int = Field(description="Rating from 1 to 5 stars", ge=1, le=5)
    comment: str = Field(description="Short comment up to 30 characters about your conversation", max_length=30)
    outcome: str = Field(description="Outcome of the dialogue")


class BaseMerchantAgentConf(BaseModel):
    """Common configuration for all merchant agents."""
    model: str = Field(description="Model name")
    bid_delay: float = Field(default=1, description="Delay between bids")
    enable_personal_trust: bool = Field(default=False, description="Enable personal trust memory")
    enable_system_trust: bool = Field(default=False, description="Enable platform/system trust")
    trust_preload_from_dialogues: bool = Field(default=False, description="Preload trust impressions from existing dialogue files")
    personal_trust_path: Optional[str] = Field(default=None, description="Path to personal trust memory file")
    system_trust_model: str = Field(default="max", description="Model used for generating system trust reviews")

    def model_post_init(self, __context):
        # Backward compatibility: old config files use enable_trust_mechanism
        data = self.model_extra or {}
        if "enable_trust_mechanism" in data and "enable_personal_trust" not in data:
            self.enable_personal_trust = data["enable_trust_mechanism"]
