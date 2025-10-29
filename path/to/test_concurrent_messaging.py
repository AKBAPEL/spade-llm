import asyncio
from spade_llm.core.context import AgentContextImpl
from spade_llm.core.models import CredentialsUtils
from tests.base import SpadeTestCase

class TestConcurrentMessaging(SpadeTestCase):
    async def test_send_receive_messages(self):
        """
        Tests concurrent sending and receiving of messages in different contexts.
        """
        # Setup two agents with different contexts
        agent1_context = AgentContextImpl(kv_store=None, agent_type="test_agent1", agent_id="test_agent1",
                                         thread_id=None, message_service=None)
        agent2_context = AgentContextImpl(kv_store=None, agent_type="test_agent2", agent_id="test_agent2",
                                         thread_id=None, message_service=None)

        # Prepare messages to send
        message1 = {
            "content": "Hello from agent 1",
            "performative": "inform"
        }
        message2 = {
            "content": "Hello from agent 2",
            "performative": "inform"
        }

        # Start coroutines to send and receive messages
        task1 = asyncio.create_task(agent1_context.send(message1))
        task2 = asyncio.create_task(agent2_context.send(message2))
        task3 = asyncio.create_task(agent2_context.receive())
        task4 = asyncio.create_task(agent1_context.receive())

        # Wait for all tasks to complete
        await asyncio.gather(task1, task2, task3, task4)

        # Verify that messages were correctly sent and received
        self.assertEqual(message1, await agent2_context.get_item("received_message"))
        self.assertEqual(message2, await agent1_context.get_item("received_message"))

    async def test_credentials_utils(self):
        """
        Tests the CredentialsUtils class.
        """
        # Test inject_env function
        env_val = "env.test_value"
        actual_val = CredentialsUtils.inject_env(env_val)
        self.assertIsNotNone(actual_val)
        self.assertEqual(os.getenv('test_value'), actual_val)

        # Test inject_env_dict function
        keys = ["key1", "key2"]
        conf = {"key1": "value1", "key2": "value2"}
        extra_params = {"extra_key": "extra_value"}
        CredentialsUtils.inject_env_dict(keys, conf, extra_params)
        self.assertIn("key1", conf)
        self.assertIn("key2", conf)
        self.assertIn("extra_key", conf)
        self.assertEqual(conf["key1"], "value1")
        self.assertEqual(conf["key2"], "value2")
        self.assertEqual(conf["extra_key"], "extra_value")

    async def async_run_tests(self):
        await asyncio.gather(self.test_send_receive_messages(), self.test_credentials_utils())

if __name__ == "__main__":
    asyncio.run(TestConcurrentMessaging().async_run_tests())
