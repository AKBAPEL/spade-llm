from spade_llm.core.agent import Agent
from spade_llm.core.behaviors import MessageHandlingBehavior, MessageTemplate
from spade_llm.core.conf import configuration, Configurable, EmptyConfig


@configuration(EmptyConfig)
class EchoAgentHandler(Agent, Configurable[EmptyConfig]):


    class EchoBehaviour(MessageHandlingBehavior):
        async def step(self):
            print(f"\nEchoAgent received message: {self.message}")

    def setup(self):
        self.add_behaviour(self.EchoBehaviour(MessageTemplate()))