# Thin re-export module for backward compatibility with config.yaml
from spade_llm.demo.platform.auction.board_agent import ProposalBoardAgent
from spade_llm.demo.platform.auction.merchant_agents import (
    FirstMerchantAgent,
    SecondMerchantAgent,
    ThirdMerchantAgent,
)

__all__ = [
    "ProposalBoardAgent",
    "FirstMerchantAgent",
    "SecondMerchantAgent",
    "ThirdMerchantAgent",
]
