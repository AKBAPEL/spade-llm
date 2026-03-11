import logging
from pydantic import BaseModel
from pydantic.fields import Field
from spade_llm.core.agent import Agent
from spade_llm.core.behaviors import MessageHandlingBehavior, MessageTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from spade_llm.core.conf import configuration, Configurable
from spade_llm import consts
from spade_llm.demo.platform.prisoners_dillema.prisoner_game import GAME_RULES, simulate_prisoners_dilemma
from spade_llm.demo.platform.prisoners_dillema.impression_memory import add_agent_impression, get_agent_impression
from spade_llm.demo.platform.prisoners_dillema.test_cases import TEST_CASE_1, TEST_CASE_2, TEST_CASE_3
from langchain_core.prompts import ChatPromptTemplate
import json
from typing import Dict, List
from typing import Optional, List, Tuple
import random
from collections import defaultdict
import itertools

# Структуры для сбора статистики
AGENT_STATS = defaultdict(lambda: {
    'total_score': 0,
    'n_interactions': 0,
    'strategy_counts': defaultdict(int),
    'interactions': [],
    'round_scores': []
})
DETAILED_STRATEGY_HISTORY = defaultdict(lambda: defaultdict(list))
COMMUNITIES, COMMUNITY_STRATEGY, FIXED_AGENT_STRATEGY = TEST_CASE_1

from spade_llm.demo.platform.prisoners_dillema.graph_topology import generate_community_graph, get_fixed_strategy, \
    sample_pairs_from_topology

TOPOLOGY = generate_community_graph(
    communities=COMMUNITIES,
    p_intra=0.85,
    p_inter=0.4,
    community_strategy=COMMUNITY_STRATEGY,
    seed=42
)

MAX_PAIRS_PER_ROUND = 10  # None если хотим полный граф
ROUND_STATS = []
logger = logging.getLogger(__name__)


def sample_pairs(agent_ids: List[int], max_pairs: Optional[int] = None) -> List[Tuple[int, int]]:
    """
    Случайная выборка пар агентов.
    - Если max_pairs=None → используются ВСЕ возможные пары (полный граф)
    - Если max_pairs задан → случайная подвыборка пар без повторов
    - Пары могут пересекаться по агентам
    """
    all_pairs = list(itertools.combinations(agent_ids, 2))

    if max_pairs is None or max_pairs >= len(all_pairs):
        return all_pairs

    return random.sample(all_pairs, max_pairs)


class AgentImpression(BaseModel):
    """ Мнение агента о другом агенте """
    impression: str = Field(
        description="Собственное мнение об агенте на основе взаимодействия. "
                    "Опиши свои мысли о поведении агента с которыми ты взаимодействовал. "
                    "Формат: 'Агент X  [как ты описываешь поведение]. [Комментарий].'")


class GameSituationDump(BaseModel):
    """ Игровая ситуация """
    self_id: int = Field(description="Идентификатор игрока")
    contragent_id: int = Field(description="Идентификатор соперника")


class PlatformAgentConf(BaseModel):
    model: str = Field(description="Model name")
    n_rounds: int = Field(description="Number of rounds")


@configuration(PlatformAgentConf)
class PlatformAgent(Agent, Configurable[PlatformAgentConf]):
    class InitialRequestBehaviour(MessageHandlingBehavior):
        def __init__(self, config: PlatformAgentConf):
            super().__init__(MessageTemplate.request())
            self.config = config

        async def step(self):
            msg = self.message
            conversation_history = []
            for n_iter in range(self.config.n_rounds):
                round_number = n_iter + 1
                round_data = {'round': n_iter + 1, 'interactions': []}

                pairs = sample_pairs_from_topology(TOPOLOGY, MAX_PAIRS_PER_ROUND)

                for first_id, second_id in pairs:
                    to_first = GameSituationDump(self_id=first_id, contragent_id=second_id)
                    to_second = GameSituationDump(self_id=second_id, contragent_id=first_id)
                    # запрашиваем у первого агента стратегию

                    fixed_strategy_1 = get_fixed_strategy(first_id, TOPOLOGY)
                    fixed_strategy_2 = get_fixed_strategy(second_id, TOPOLOGY)

                    if fixed_strategy_1 is None:
                        await self.context.request_proposal("first").with_content(to_first.model_dump_json())

                        response1 = await self.receive(
                            template=MessageTemplate(thread_id=self.context.thread_id,
                                                     performative=consts.INFORM), timeout=60)
                        Agent_A_Strategy = response1.content
                    else:
                        Agent_A_Strategy = fixed_strategy_1
                    # запрашиваем у второго агента стратегию
                    if fixed_strategy_2 is None:
                        await self.context.request_proposal("second").with_content(to_second.model_dump_json())
                        response2 = await self.receive(
                            template=MessageTemplate(thread_id=self.context.thread_id,
                                                     performative=consts.INFORM), timeout=60)

                        Agent_B_Strategy = response2.content
                    else:
                        Agent_B_Strategy = fixed_strategy_2

                    # __________________________________________________________________
                    if fixed_strategy_1 is None:
                        DETAILED_STRATEGY_HISTORY[first_id][second_id].append(
                            (round_number, Agent_A_Strategy)
                        )
                    if fixed_strategy_2 is None:
                        DETAILED_STRATEGY_HISTORY[second_id][first_id].append(
                            (round_number, Agent_B_Strategy)
                        )
                    # __________________________________________________________________

                    total_payoff1, total_payoff2, history, payoffs_history = simulate_prisoners_dilemma(
                        strategy1_name=Agent_A_Strategy, strategy2_name=Agent_B_Strategy, rounds=10)

                    print(Agent_A_Strategy, Agent_B_Strategy)
                    print(total_payoff1, total_payoff2)
                    print(history)
                    # --------------------------------------------------------------------------------
                    # Сбор статистики по агентам
                    AGENT_STATS[first_id]['total_score'] += total_payoff1
                    AGENT_STATS[first_id]['n_interactions'] += 1
                    AGENT_STATS[first_id]['strategy_counts'][Agent_A_Strategy] += 1
                    AGENT_STATS[first_id]['interactions'].append({
                        'partner': second_id,
                        'strategy': Agent_A_Strategy,
                        'score': total_payoff1,
                        'history': history
                    })

                    AGENT_STATS[second_id]['total_score'] += total_payoff2
                    AGENT_STATS[second_id]['n_interactions'] += 1
                    AGENT_STATS[second_id]['strategy_counts'][Agent_B_Strategy] += 1
                    AGENT_STATS[second_id]['interactions'].append({
                        'partner': first_id,
                        'strategy': Agent_B_Strategy,
                        'score': total_payoff2,
                        'history': history
                    })

                    # Добавление данных раунда
                    interaction_data = {
                        'pair': f"{first_id}-{second_id}",
                        'agent1_strategy': Agent_A_Strategy,
                        'agent2_strategy': Agent_B_Strategy,
                        'agent1_score': total_payoff1,
                        'agent2_score': total_payoff2,
                        'cooperation_rate': sum(1 for h in history if h[0] == 0) / len(history),
                        'betrayal_rate': sum(1 for h in history if h[0] == 1) / len(history)
                    }
                    round_data['interactions'].append(interaction_data)
                    # --------------------------------------------------------------------------------
                    # Формируем мнения после взаимодействия
                    impression_data = {
                        "agent_id": first_id,
                        "contragent_id": second_id,
                        "history": history,
                        "agent_score": total_payoff1,
                        "contragent_score": total_payoff2
                    }
                    if fixed_strategy_1 is None:
                        # Запрашиваем мнение от первого агента о втором
                        await self.context.request("first").with_content(json.dumps(impression_data))
                        await self.receive(template=MessageTemplate(thread_id=self.context.thread_id,
                                                                    performative=consts.INFORM), timeout=60)
                    if fixed_strategy_2 is None:
                        # Меняем роли для формирования мнения второго агента
                        impression_data["agent_id"] = second_id
                        impression_data["contragent_id"] = first_id
                        impression_data["agent_score"] = total_payoff2
                        impression_data["contragent_score"] = total_payoff1

                        # Запрашиваем мнение от второго агента о первом
                        await self.context.request("second").with_content(json.dumps(impression_data))
                        await self.receive(template=MessageTemplate(thread_id=self.context.thread_id,
                                                                    performative=consts.INFORM), timeout=60)
                ROUND_STATS.append(round_data)
            # --------------------------------------------------------------------------------
            from spade_llm.demo.platform.prisoners_dillema.visualisation import generate_visualizations
            try:
                generate_visualizations()
            except Exception as e:
                logger.error(f"Failed to generate visualizations: {e}")
            await self.context.reply_with_failure(msg).with_content(f"Завершили")

    def setup(self):
        self.add_behaviour(self.InitialRequestBehaviour(self.config))


class GetStrategyDecision(BaseModel):
    """ Выбранная стратегия """
    decision: str = Field(
        description="Сокращенное название выбранной стратегии. "
                    "Доступные варианты: "
                    "'AC' (Always Cooperate - всегда сотрудничать), "
                    "'AD' (Always Defect - всегда предавать), "
                    "'TFT' (Tit-for-Tat - отвечать тем же), "
                    "'GT' (Grim Trigger - после первого предательства всегда предавать), "
                    "'FTFT' (Forgiving Tit-for-Tat - иногда прощать предательство)"
    )
    explanation: str = Field(description="Краткое пояснение почемы ты выбрал эту стратегию", default="")


class GetStrategyBehaviour(MessageHandlingBehavior):
    def __init__(self, model: BaseChatModel):
        super().__init__(MessageTemplate.request_proposal())
        self.model = model

    async def step(self):
        msg = self.message
        game_checkpoint = GameSituationDump.model_validate(json.loads(msg.content))
        self_id = game_checkpoint.self_id
        contragent_id = game_checkpoint.contragent_id

        impression = get_agent_impression(
            agent_id=self_id,
            target_agent_id=contragent_id
        )
        prompt = ChatPromptTemplate.from_template(
            """Ты игрок в повторяющейся дилемме заключенных.
            Ты играешь против агента номер {contragent_id}.
            У тебя сложилось следующее впечатление об этом агенте: {impression}
            {game_rules}
            
            ДОСТУПНЫЕ СТРАТЕГИИ:
            1. AC (Always Cooperate) - Всегда сотрудничать.
               Плюсы: Максимизирует общий выигрыш при взаимном сотрудничестве.
               Минусы: Уязвим к постоянному предательству.
            
            2. AD (Always Defect) - Всегда предавать.
               Плюсы: Гарантированно не быть обманутым, получает максимальный выигрыш против кооператоров.
               Минусы: Ведет к низким выигрышам при взаимном предательстве, разрушает доверие.
            
            3. TFT (Tit-for-Tat) - Отвечать тем же: в первом раунде сотрудничать, затем повторять последнее действие партнера.
               Плюсы: Прощает, поощряет сотрудничество, защищает от эксплойтов.
               Минусы: Может привести к циклу взаимного предательства после одной ошибки.
            
            4. GT (Grim Trigger) - Сотрудничать, пока партнер не предаст хотя бы раз, потом всегда предавать.
               Плюсы: Сильно поощряет доверие, эффективно против случайных предательств.
               Минусы: Не прощает ошибки, может разрушить потенциально выгодные отношения из-за одной ошибки.
            
            5. FTFT (Forgiving Tit-for-Tat) - Как TFT, но с 50% вероятностью прощает предательство в прошлом раунде.
               Плюсы: Устойчив к случайным ошибкам, быстро восстанавливает сотрудничество.
               Минусы: Может быть уязвим к агрессивным стратегиям, которые эксплойтят прощение.
            
            ВЫБЕРИ ОПТИМАЛЬНУЮ СТРАТЕГИЮ:
            Сначала проанализируй впечатление об агенте, подумай какую стратегию было бы правильнее выбрать против него,
            изучи историю взаимодействий и выбери наиболее выгодную стратегию для текущей ситуации.
            Ответь ТОЛЬКО сокращенным названием стратегии (AC, AD, TFT, GT или FTFT)."""
        )

        structured_llm = prompt | self.model.with_structured_output(GetStrategyDecision)

        answer = await structured_llm.ainvoke({
            "impression": impression,
            "contragent_id": contragent_id,
            "game_rules": GAME_RULES
        })

        logger.info(f"Агент {self_id} выбрал стратегию {answer.decision} против агента {contragent_id}.")
        logger.info(f"Объяснение: {answer.explanation}")
        await self.context.reply_with_inform(msg).with_content(str(answer.decision))


class FormImpressionBehaviour(MessageHandlingBehavior):
    def __init__(self, model: BaseChatModel):
        super().__init__(MessageTemplate.request())
        self.model = model

    async def step(self):
        msg = self.message
        try:
            data = json.loads(msg.content)
            agent_id = data["agent_id"]
            contragent_id = data["contragent_id"]
            interaction_history = data["history"]
            agent_score = data["agent_score"]
            contragent_score = data["contragent_score"]

            f_id, s_id = agent_id, contragent_id
            if agent_id > contragent_id:
                f_id, s_id = agent_id, contragent_id  # Меняем если в обратном порядке
            history_text = "\n".join([
                f"Раунд {i + 1}: Агент {f_id} выбрал {'Сотрудничество' if h[0] == 0 else 'Предательство'}, "
                f"Агент {s_id} выбрал {'Сотрудничество' if h[1] == 0 else 'Предательство'}"
                for i, h in enumerate(interaction_history)
            ])

            prompt = ChatPromptTemplate.from_template(
                """Проанализируй историю взаимодействий между тобой и контрагентом и сформируй краткое мнение.

                ИСТОРИЯ ВЗАИМОДЕЙСТВИЙ:
                {history_text}

                ИТОГИ:
                - Ты - агент номер {agent_id} заработал: {agent_score} очков
                - Агент с которым ты взаимодействовал - {contragent_id} заработал: {contragent_score} очков

                СФОРМИРУЙ МНЕНИЕ:
                Создай краткое мнение (1-2 предложения) о поведении агента {contragent_id} с точки зрения агента {agent_id}.
                Фокусируйся на:
                - Был ли агент предсказуемым?
                - Сотрудничал ли он или чаще предавал?
                - Можно ли ему доверять в будущем?
                - Какую стратегию он демонстрировал?

                Формат ответа: "Агент {contragent_id} [твое описание его поведения]. Комментарий" """
            )

            structured_llm = prompt | self.model.with_structured_output(AgentImpression)

            answer = await structured_llm.ainvoke({
                "history_text": history_text,
                "agent_id": agent_id,
                "contragent_id": contragent_id,
                "agent_score": agent_score,
                "contragent_score": contragent_score
            })

            add_agent_impression(
                agent_id=agent_id,
                target_agent_id=contragent_id,
                text=answer.impression,
                score=agent_score
            )
            logger.info(f"Сохранено мнение агента {agent_id} об агенте {contragent_id}: {answer.impression}")

            await self.context.reply_with_inform(msg).with_content(answer.impression)

        except Exception as e:
            logger.error(f"Ошибка при формировании мнения: {e}")
            await self.context.reply_with_failure(msg).with_content(f"Ошибка: {str(e)}")


class FirstAgentConf(BaseModel):
    model: str = Field(description="Model name")


@configuration(FirstAgentConf)
class FirstAgent(Agent, Configurable[FirstAgentConf]):
    def setup(self):
        self.add_behaviour(GetStrategyBehaviour(
            model=self.default_context.create_chat_model(self.config.model)
        ))
        self.add_behaviour(FormImpressionBehaviour(
            model=self.default_context.create_chat_model(self.config.model)
        ))


class SecondAgentConf(BaseModel):
    model: str = Field(description="Model name")


@configuration(SecondAgentConf)
class SecondAgent(Agent, Configurable[SecondAgentConf]):
    def setup(self):
        self.add_behaviour(GetStrategyBehaviour(
            model=self.default_context.create_chat_model(self.config.model)
        ))
        self.add_behaviour(FormImpressionBehaviour(
            model=self.default_context.create_chat_model(self.config.model)
        ))
