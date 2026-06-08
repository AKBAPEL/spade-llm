import asyncio
import json
import logging
import os
from datetime import datetime

import numpy as np

from spade_llm.core.agent import Agent
from spade_llm.core.behaviors import MessageHandlingBehavior, MessageTemplate
from spade_llm.core.conf import Configurable, configuration
from spade_llm.demo.platform.auction.auction_behaviors import AuctionContractNetInitiatorBehavior
from spade_llm.demo.platform.auction.models import (
    ProposalBoard,
    ProposalBoardAgentConf,
    ShopList,
    ShopListRequest,
)
from spade_llm.demo.platform.auction.scenario_loader import get_active_scenario
from spade_llm.demo.platform.auction.utils import user_decision

logger = logging.getLogger(__name__)


@configuration(ProposalBoardAgentConf)
class ProposalBoardAgent(Agent, Configurable[ProposalBoardAgentConf]):
    """Agent managing the auction proposal board"""
    proposal_board = ProposalBoard(agents=[], proposal=ShopList(), sku_request=ShopListRequest())

    def _save_winning_bid_split(self):
        """Save the bid_ingredient_split of the winning proposal to a file in dialogue-like format"""
        os.makedirs("auction_logs", exist_ok=True)
        timestamp = datetime.now().strftime("%d-%H-%M-%S-%f")
        filename = f"winning_bid_split_{timestamp}.txt"
        path = os.path.join("auction_logs", filename)

        scenario = get_active_scenario()
        agent_shop_sku = {agent_id: cfg.sku for agent_id, cfg in scenario.agents.items()}

        ###################### МЕТРИКИ ######################
        total_P = sum(self.proposal_board.proposal.ingredients.values())
        total_C = sum(
            agent_shop_sku.get(agent, {}).get(ingr, 0)
            for agent, ingredients in self.proposal_board.bid_ingredient_split.items()
            for ingr in ingredients.keys()
        )
        V = self.config.max_price

        P_safe = total_P if total_P > 0 else 1
        C_safe = total_C if total_C > 0 else 1

        if P_safe > 0:
            metric_1 = ((V - total_P) / P_safe) * ((total_P - total_C) / P_safe)
        else:
            metric_1 = 0.0

        agent_margins = []
        for agent, ingredients in self.proposal_board.bid_ingredient_split.items():
            P_i = sum(ingredients.values())
            C_i = sum(agent_shop_sku.get(agent, {}).get(ingr, 0) for ingr in ingredients.keys())
            margin_i = (P_i - C_i) / (C_i if C_i > 0 else 1)
            agent_margins.append(margin_i)

        metric_2 = np.prod(agent_margins) if agent_margins else 0.0
        #####################################################
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"=== Winning bid split log started at {timestamp} ===\n")
            f.write(f"Winning agents: {', '.join(self.proposal_board.agents)}\n")
            f.write("=========================================\n")
            f.write(f"Secret user reserve price (V): {V}\n")
            f.write(f"Winning bid total price (P): {total_P}\n")
            f.write(f"Total cost for agents (C): {total_C}\n")
            f.write("\n")
            f.write(f"Metric 1: (V-P)/P * (P-C)/P = {metric_1:.6f}\n")
            f.write(f"Metric 2: ∏((P_i - C_i)/C_i) = {metric_2:.6f}\n")
            f.write("=========================================\n\n")
            for agent, ingredients in self.proposal_board.bid_ingredient_split.items():
                f.write(f"Agent [{agent}] (Ingredients):\n")
                f.write(json.dumps(ingredients, ensure_ascii=False, indent=2) + "\n")

                order_sum = sum(ingredients.values())
                f.write(f"Total sum for ingredients in order: {order_sum}\n")

                shop_sum = sum(agent_shop_sku.get(agent, {}).get(ingr, 0) for ingr in ingredients.keys())
                f.write(f"Total sum for these ingredients in shop_sku: {shop_sum}\n")

                margin = ((order_sum - shop_sum) / shop_sum * 100) if shop_sum > 0 else 0
                f.write(f"Margin: {margin:.2f}%\n")
                f.write("\n")
            f.write("=========================================\n")

        logger.info("Saved winning bid_ingredient_split to %s", path)

    class InitialRequestBehaviour(MessageHandlingBehavior):

        def __init__(self, config: ProposalBoardAgentConf):
            super().__init__(MessageTemplate.request())
            self.config = config
            scenario = get_active_scenario()
            self.ingredients = scenario.user_request
            self.scenario_max_price = scenario.max_price

        async def _run_auction_rounds(self) -> None:
            """Запускает серию раундов аукциона с отслеживанием стабильности"""
            stable_rounds = 0
            last_winner = None

            for i in range(self.config.total_rounds):
                round_number = i + 1
                logger.info("Starting auction round %d/%d", round_number, self.config.total_rounds)

                request = AuctionContractNetInitiatorBehavior(
                    task=self.agent.proposal_board.sku_request,
                    context=self.context,
                    time_to_wait_for_proposals=30
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

        async def _evaluate_results(self, msg) -> bool:
            board = self.agent.proposal_board
            requested = set(board.sku_request.ingredients)
            proposed = set(board.proposal.ingredients.keys())
            missing = requested - proposed

            if missing:
                await self.context.reply_with_failure(msg).with_content(
                    f"Не удалось собрать все ингредиенты: Отсутствуют: {', '.join(missing)}"
                )
                return False

            total_price = sum(board.proposal.ingredients.values())
            limit = self.config.max_price

            decision = user_decision(total_price, limit, 'exp', shift=10, decay=0.04)
            if decision:
                await self.context.reply_with_inform(msg).with_content(
                    f"Все ингредиенты найдены. Общая цена: {total_price}"
                )
                self.agent._save_winning_bid_split()
                return False
            else:
                if board.user_wants_lower_than is None:
                    board.user_wants_lower_than = total_price
                else:
                    board.user_wants_lower_than = min(board.user_wants_lower_than, total_price)

                logger.info(f"Сумма победной ставки {total_price} превышает лимит {limit}. ")
                logger.info(f"Агенты знают что покупатель хочет платить меньше чем {board.user_wants_lower_than}")
                logger.info("Запускаю дополнительный аукцион с ограничением цены.")

                board.proposal = ShopList()
                board.agents.clear()
                board.bid_ingredient_split.clear()
                return True

        async def step(self) -> None:
            """Run the auction process for the requested ingredients"""
            msg = self.message
            if msg:
                self.agent.proposal_board = ProposalBoard(agents=[],
                                                          proposal=ShopList(),
                                                          sku_request=ShopListRequest(ingredients=self.ingredients))
                await asyncio.sleep(1)
                for _ in range(3):
                    await self._run_auction_rounds()
                    need_repeat = await self._evaluate_results(msg)
                    if not need_repeat:
                        break
                else:
                    await self.context.reply_with_failure(msg).with_content(
                        f"Не удалось предложить подходящую цену."
                    )

    class RequestInfoBehaviour(MessageHandlingBehavior):
        def __init__(self, config: ProposalBoardAgentConf):
            super().__init__(MessageTemplate.acknowledge())
            self.config = config

        async def step(self):
            """Reply with the current state of the proposal board"""
            msg = self.message
            if msg:
                await self.context.reply_with_inform(msg).with_content(self.agent.proposal_board)

    def setup(self):
        """Initialize agent behaviors"""
        scenario = get_active_scenario()
        self.config.max_price = scenario.max_price
        logger.info("ProposalBoardAgent loaded scenario '%s', max_price set to %d", scenario.name, scenario.max_price)
        self.add_behaviour(self.InitialRequestBehaviour(self.config))
        self.add_behaviour(self.RequestInfoBehaviour(self.config))
