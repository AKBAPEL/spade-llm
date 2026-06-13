import logging
import time
from typing import List, Optional, Set

from spade_llm import consts
from spade_llm.core.behaviors import ContextBehaviour, MessageTemplate
from spade_llm.demo.platform.auction.models import AuctionProposal, ShopList, ShopListRequest
from spade_llm.demo.platform.contractnet.discovery import AgentDescription, AgentSearchRequest, AgentSearchResponse, DF_ADDRESS

logger = logging.getLogger(__name__)


class AuctionContractNetInitiatorBehavior(ContextBehaviour):
    """Behavior for initiating the contract net protocol in the auction"""
    task: ShopListRequest
    proposal: Optional[ShopList] = None
    result: Optional[MessageTemplate]
    time_to_wait_for_proposals: float
    _started_at: float

    def __init__(self, task: ShopListRequest, context, time_to_wait_for_proposals: float = 30):
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
        if task is None:
            logger.error('NONE 1')
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
            if self.task is None:
                logger.error('NONE 2')
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

    def _has_low_system_rating(self, proposal: AuctionProposal) -> bool:
        """Check if any author of the proposal has a low system trust rating."""
        config = self.agent.config
        if not getattr(config, "system_trust_enabled", False):
            return False
        system_trust = self.agent.system_trust
        threshold = getattr(config, "trust_reject_threshold", 2.0)
        min_reviews = getattr(config, "min_reviews_for_reject", 5)
        for author in proposal.authors:
            if system_trust.is_low_score(author, threshold=threshold, min_reviews=min_reviews):
                logger.info(
                    "Rejecting proposal from %s: agent %s has low system rating",
                    proposal.authors, author,
                )
                return True
        return False

    async def extract_winner_and_notify_losers(self, proposals: List[tuple[str, AuctionProposal]]) -> Optional[ShopList]:
        """Select the winning proposal and notify losers"""
        if not proposals:
            return None

        winner_sender = None
        winner = None
        for sender, proposal in proposals:
            if winner is not None:
                logger.info("Already updated board. Refusing")
                await self.context.refuse(sender).with_content('UPDATING')
                continue

            if self._has_low_system_rating(proposal):
                logger.info("Proposal from %s rejected due to low system trust rating", proposal.authors)
                await self.context.refuse(sender).with_content(
                    "Ставка отклонена: один или несколько авторов имеют низкий системный рейтинг доверия."
                )
                continue

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
                self.agent.proposal_board.bid_ingredient_split = winner.bid_ingredient_split
                await self.context.accept(sender).with_content('')
            else:
                logger.info("Not better, rejecting")
                await self.context.refuse(sender).with_content('')
        return winner.prop if winner else self.agent.proposal_board.proposal
