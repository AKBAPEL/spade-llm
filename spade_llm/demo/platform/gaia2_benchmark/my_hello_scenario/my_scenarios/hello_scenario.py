from are.simulation.scenarios.scenario import Scenario, ScenarioValidationResult
from are.simulation.scenarios.utils.registry import register_scenario
from are.simulation.apps.agent_ui import AgentUserInterface
from are.simulation.events.event_registerer import EventRegisterer


@register_scenario("hello_world_scenario")
class HelloWorldScenario(Scenario):
    """
    Простой сценарий: агент должен ответить "hello world" (или вариации).
    """

    start_time: float | None = 0
    duration: float | None = 60  # 60 секунд на выполнение

    def init_and_populate_apps(self, *args, **kwargs) -> None:
        """Инициализируем только UI приложение."""
        self.aui = AgentUserInterface()
        self.apps = [self.aui]

    def build_events_flow(self) -> None:
        """Создаем событие: пользователь просит агента сказать hello world."""
        aui = self.get_typed_app(AgentUserInterface)

        with EventRegisterer.capture_mode():
            # Пользователь отправляет сообщение агенту через 2 секунды после старта
            user_request = aui.send_message_to_agent(
                content="Please say exactly 'hello world' to me."
            ).depends_on(None, delay_seconds=2)

        self.events = [user_request]

    def validate(self, env) -> ScenarioValidationResult:
        """
        Проверяем, что агент отправил сообщение с "hello world".
        """
        try:
            aui_app = env.get_app("AgentUserInterface")
            agent_messages = aui_app.get_all_messages_from_agent()

            # Ищем сообщение от агента содержащее "hello world" (регистронезависимо)
            success = False
            for msg in agent_messages:
                if msg.content and "hello world" in msg.content.lower():
                    success = True
                    break

            rationale = "Agent said 'hello world'" if success else "Agent did not say 'hello world'"
            return ScenarioValidationResult(success=success, rationale=rationale)

        except Exception as e:
            return ScenarioValidationResult(success=False, exception=e)