import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from langchain_core.language_models.chat_models import BaseChatModel

from spade_llm import consts
from spade_llm.core.api import AgentContext
from spade_llm.core.behaviors import ContextBehaviour, MessageTemplate
from spade_llm.demo.platform.auction.models import (
    SystemTrustReview,
    TrustImpressionRecord,
    TrustScoreRequest,
    TrustScoreResponse,
)

logger = logging.getLogger(__name__)


class PersonalTrustMechanism:
    """Personal trust memory stored locally per merchant agent."""

    def __init__(self, agent_id: str, path: Optional[str] = None):
        self.agent_id = agent_id
        self.path = path or f"data/memory/{agent_id}_trust_memory.json"
        self.memory: Dict[str, List[TrustImpressionRecord]] = {}
        self.load()

    def load(self):
        """Load trust memory from disk, migrating legacy flat format."""
        try:
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                migrated: Dict[str, List[TrustImpressionRecord]] = {}
                for partner, records in raw.items():
                    if isinstance(records, str):
                        migrated[partner] = [
                            TrustImpressionRecord(
                                timestamp=datetime.now().isoformat(),
                                impression=records,
                                outcome="unknown",
                                source="legacy",
                            )
                        ]
                    elif isinstance(records, list):
                        migrated[partner] = [
                            TrustImpressionRecord.model_validate(r) for r in records
                        ]
                    else:
                        migrated[partner] = []
                self.memory = migrated
                total = sum(len(v) for v in self.memory.values())
                logger.info(
                    "Loaded personal trust for %s: %d partners, %d records",
                    self.agent_id, len(self.memory), total,
                )
            else:
                self.memory = {}
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load personal trust for %s: %s", self.agent_id, e)
            self.memory = {}

    def save(self):
        """Persist trust memory to disk atomically."""
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp_path = self.path + ".tmp"
            serializable = {
                partner: [r.model_dump() for r in records]
                for partner, records in self.memory.items()
            }
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(serializable, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self.path)
            total = sum(len(v) for v in self.memory.values())
            logger.info(
                "Saved personal trust for %s: %d partners, %d records",
                self.agent_id, len(self.memory), total,
            )
        except OSError as e:
            logger.warning("Failed to save personal trust for %s: %s", self.agent_id, e)

    def add_impression(self, partner_id: str, impression: str, outcome: str, source: str = "runtime"):
        """Append a new trust impression record for a partner."""
        record = TrustImpressionRecord(
            timestamp=datetime.now().isoformat(),
            impression=impression,
            outcome=outcome,
            source=source,
        )
        if partner_id not in self.memory:
            self.memory[partner_id] = []
        self.memory[partner_id].append(record)

    def get_latest_impression(self, partner_id: str) -> Optional[str]:
        """Return the latest impression text for a partner, or None."""
        records = self.memory.get(partner_id, [])
        if not records:
            return None
        return records[-1].impression

    def get_partner_summary(self, partner_id: str) -> str:
        """Return a short summary of personal trust for a partner."""
        records = self.memory.get(partner_id, [])
        if not records:
            return "Личного опыта нет."
        lines = [f"Личных впечатлений: {len(records)}. Последние:"]
        for r in records[-3:]:
            lines.append(f"  [{r.outcome}] {r.impression}")
        return "\n".join(lines)

    def build_context(self, partner_ids: List[str], self_id: str) -> str:
        """Build a context string from trust memory for available partners."""
        lines = ["Твои прошлые личные впечатления о партнёрах (от новых к старым):"]
        has_any = False
        for partner in partner_ids:
            if partner == self_id:
                continue
            records = self.memory.get(partner, [])
            if records:
                latest = records[-1]
                lines.append(f"- {partner} ({latest.timestamp}): {latest.impression}")
                has_any = True
            else:
                lines.append(f"- {partner}: Прошлого опыта взаимодействия нет.")
        if not has_any:
            return "Прошлого личного опыта взаимодействия ни с одним агентом нет."
        return "\n".join(lines)


class SystemTrustMechanism:
    """Platform-wide trust board maintained by the ProposalBoardAgent."""

    def __init__(self):
        self._reviews: Dict[str, List[SystemTrustReview]] = {}

    def add_review(self, review: SystemTrustReview):
        """Add a review to the board and assign timestamp if missing."""
        if review.timestamp is None:
            review.timestamp = datetime.now().isoformat()
        target = review.target_agent
        if target not in self._reviews:
            self._reviews[target] = []
        self._reviews[target].append(review)

    def get_reviews(self, target_agent: str) -> List[SystemTrustReview]:
        """Return all reviews for an agent, newest first."""
        return sorted(
            self._reviews.get(target_agent, []),
            key=lambda r: r.timestamp or "",
            reverse=True,
        )

    def get_average_rating(self, target_agent: str) -> float:
        """Return average rating for an agent, or 0.0 if no reviews."""
        reviews = self._reviews.get(target_agent, [])
        if not reviews:
            return 0.0
        return sum(r.rating for r in reviews) / len(reviews)

    def get_counts(self, target_agent: str) -> Tuple[int, int, int]:
        """Return (positive_count, negative_count, total_count) for an agent."""
        reviews = self._reviews.get(target_agent, [])
        positive = sum(1 for r in reviews if r.rating >= 4)
        negative = sum(1 for r in reviews if r.rating <= 3)
        return positive, negative, len(reviews)

    def get_review_slice(
        self,
        target_agent: str,
        limit: int = 5,
    ) -> Tuple[List[SystemTrustReview], List[SystemTrustReview]]:
        """Return up to `limit` most recent positive (4-5★) and negative (1-3★) reviews."""
        reviews = self.get_reviews(target_agent)
        positive = [r for r in reviews if r.rating >= 4][:limit]
        negative = [r for r in reviews if r.rating <= 3][:limit]
        return positive, negative

    def is_low_score(
        self,
        target_agent: str,
        threshold: float = 2.0,
        min_reviews: int = 5,
    ) -> bool:
        """Check if agent's average rating is below threshold with enough reviews."""
        reviews = self._reviews.get(target_agent, [])
        if len(reviews) < min_reviews:
            return False
        return self.get_average_rating(target_agent) < threshold

    def serialize(self) -> dict:
        """Serialize board to a plain dict."""
        return {
            target: [r.model_dump() for r in reviews]
            for target, reviews in self._reviews.items()
        }

    def deserialize(self, data: dict):
        """Load board from a plain dict."""
        self._reviews = {
            target: [SystemTrustReview.model_validate(r) for r in reviews]
            for target, reviews in data.items()
        }

    def save(self, path: str):
        """Persist board to disk atomically."""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp_path = path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self.serialize(), f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, path)
            logger.info("Saved system trust board to %s", path)
        except OSError as e:
            logger.warning("Failed to save system trust board to %s: %s", path, e)

    def load(self, path: str):
        """Load board from disk if it exists."""
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.deserialize(data)
                logger.info("Loaded system trust board from %s", path)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load system trust board from %s: %s", path, e)


def build_trust_score_response(
    system_trust: SystemTrustMechanism,
    target_agent: str,
) -> TrustScoreResponse:
    """Build a TrustScoreResponse for a target agent."""
    avg = system_trust.get_average_rating(target_agent)
    positive_count, negative_count, total = system_trust.get_counts(target_agent)
    positive, negative = system_trust.get_review_slice(target_agent)
    return TrustScoreResponse(
        target_agent=target_agent,
        average_rating=avg,
        total_reviews=total,
        positive_count=positive_count,
        negative_count=negative_count,
        positive_reviews=positive,
        negative_reviews=negative,
    )


class SystemTrustClient:
    """Client used by merchant agents to query the platform trust board."""

    def __init__(self, context: AgentContext, behavior: ContextBehaviour):
        self.context = context
        self.behavior = behavior

    async def get_partner_trust_score(self, target_agent: str, timeout: float = 10) -> Optional[TrustScoreResponse]:
        """Request trust score for a partner from the proposal board."""
        try:
            await self.context.trust_request("proposal_board").with_content(
                TrustScoreRequest(target_agent=target_agent).model_dump_json()
            )
            response = await self.behavior.receive(
                MessageTemplate.inform(),
                timeout=timeout,
            )
            if response is None:
                logger.warning("No trust score response for %s", target_agent)
                return None
            return TrustScoreResponse.model_validate_json(response.content)
        except Exception as e:
            logger.warning("Failed to get trust score for %s: %s", target_agent, e)
            return None


class TrustReviewBuilder:
    """Generates a system trust review after a dialogue."""

    def __init__(self, model: BaseChatModel, prompt):
        self.model = model
        self.prompt = prompt

    async def build_review(
        self,
        conversation_history: List[str],
        agent_role: str,
        partner_id: str,
        outcome: str,
        personal_impression: Optional[str],
    ) -> SystemTrustReview:
        """Generate a star rating + short comment review."""
        chain = self.prompt | self.model
        answer = await chain.ainvoke({
            "conversation_history": "\n".join(conversation_history),
            "agent_role": agent_role,
            "partner_id": partner_id,
            "outcome": outcome,
            "personal_impression": personal_impression or "Нет личного впечатления.",
        })
        cleaned = self._clean_json(answer.content)
        try:
            data = json.loads(cleaned)
            rating = int(data.get("rating", 3))
            rating = max(1, min(5, rating))
            comment = str(data.get("comment", "Средний опыт"))[:30]
            review_outcome = str(data.get("outcome", outcome))
            return SystemTrustReview(
                reviewer_id=agent_role,
                target_agent=partner_id,
                rating=rating,
                comment=comment,
                outcome=review_outcome,
            )
        except Exception as e:
            logger.warning("Failed to parse trust review, using neutral fallback: %s", e)
            return SystemTrustReview(
                reviewer_id=agent_role,
                target_agent=partner_id,
                rating=3,
                comment="Средний опыт",
                outcome=outcome,
            )

    @staticmethod
    def _clean_json(text: str) -> str:
        text = text.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start:end + 1]
        return text.strip()


class CombinedTrustContextBuilder:
    """Combines personal trust memory and system trust score into a prompt string."""

    @staticmethod
    def build_context(
        partner_id: str,
        personal: Optional[PersonalTrustMechanism],
        system_score: Optional[TrustScoreResponse],
    ) -> str:
        """Build a combined trust context string for a partner."""
        lines = [f"Информация о доверии к партнёру {partner_id}:"]

        if system_score is None:
            lines.append("Системный рейтинг: отсутствует (платформа не предоставила данных).")
        else:
            lines.append(
                f"Системный рейтинг: средняя оценка {system_score.average_rating:.2f}★ "
                f"({system_score.total_reviews} отзывов: "
                f"{system_score.positive_count} положительных, {system_score.negative_count} отрицательных)."
            )
            if system_score.positive_reviews:
                lines.append("Последние положительные отзывы (4-5★):")
                for r in system_score.positive_reviews:
                    lines.append(f"  {r.rating}★ ({r.outcome}): {r.comment}")
            if system_score.negative_reviews:
                lines.append("Последние отрицательные отзывы (1-3★):")
                for r in system_score.negative_reviews:
                    lines.append(f"  {r.rating}★ ({r.outcome}): {r.comment}")

        if personal is None:
            lines.append("Личный опыт: не используется.")
        else:
            lines.append(personal.get_partner_summary(partner_id))

        if system_score is not None:
            if system_score.average_rating >= 4.0:
                lines.append("Рекомендация: у партнёра высокий системный рейтинг — можно идти на сотрудничество.")
            elif system_score.average_rating < 2.0 and system_score.total_reviews >= 5:
                lines.append("Рекомендация: у партнёра низкий системный рейтинг — будь осторожен, жёсткие условия или отказ.")
            elif system_score.total_reviews == 0:
                lines.append("Рекомендация: системных отзывов нет — ориентируйся на личный опыт и текущую выгоду.")
            else:
                lines.append("Рекомендация: системный рейтинг нейтральный — действуй по общей логике.")

        return "\n".join(lines)
