from typing import List, Tuple, Dict, Callable
import random


def always_cooperate(history: List[Tuple[int, int]], agent_position: int, opponent_position: int) -> int:
    """Всегда сотрудничать (0)"""
    return 0


def always_defect(history: List[Tuple[int, int]], agent_position: int, opponent_position: int) -> int:
    """Всегда предавать (1)"""
    return 1


def tit_for_tat(history: List[Tuple[int, int]], agent_position: int, opponent_position: int) -> int:
    """
    Око за око (Tit-for-Tat):
    - Первый раунд: сотрудничать
    - Последующие раунды: повторять последнее действие партнера
    """
    if not history:
        return 0  # Первый раунд - сотрудничаем

    # Получаем последнее действие партнера (противоположная позиция)
    last_round = history[-1]
    partner_decision = last_round[opponent_position]
    return partner_decision


def grim_trigger(history: List[Tuple[int, int]], agent_position: int, opponent_position: int) -> int:
    """
    Грозный триггер (Grim Trigger):
    - Сотрудничает, пока партнер не предаст хотя бы один раз
    - После первого предательства партнера - всегда предает
    """
    if not history:
        return 0

    # Проверяем всю историю на наличие предательств партнера
    for round_history in history:
        partner_decision = round_history[opponent_position]
        if partner_decision == 1:  # Партнер предал хотя бы один раз
            return 1
    return 0


def forgiving_tit_for_tat(history: List[Tuple[int, int]], agent_position: int, opponent_position: int) -> int:
    """
    Прощающий око за око (Forgiving Tit-for-Tat):
    - Первый раунд: сотрудничать
    - Если партнер предал в прошлом раунде, с 50% вероятностью простить и сотрудничать
    - Иначе: повторять последнее действие партнера
    """
    if not history:
        return 0

    last_round = history[-1]
    partner_decision = last_round[opponent_position]

    if partner_decision == 1:  # Партнер предал в прошлом раунде
        # С 50% вероятностью прощаем и сотрудничаем
        return 0 if random.random() < 0.5 else 1
    else:
        return 0  # Партнер сотрудничал - сотрудничаем


def random_strategy(history: List[Tuple[int, int]], agent_position: int, opponent_position: int) -> int:
    """Случайная стратегия"""
    return random.choice([0, 1])


# Словарь стратегий с биндами
STRATEGIES: Dict[str, Callable[[List[Tuple[int, int]], int, int], int]] = {
    "AC": always_cooperate,  # Always Cooperate
    "AD": always_defect,  # Always Defect
    "TFT": tit_for_tat,  # Tit-for-Tat
    "GT": grim_trigger,  # Grim Trigger
    "FTFT": forgiving_tit_for_tat,  # Forgiving Tit-for-Tat
    "RAND": random_strategy  # Random Strategy
}


def calculate_payoffs(decision1: int, decision2: int) -> Tuple[int, int]:
    """
    Матрица выигрышей для дилеммы заключенного:
    (decision1, decision2) -> (payoff1, payoff2)

    0 = сотрудничать, 1 = предать

    |          | Партнер: 0 (C) | Партнер: 1 (D) |
    |----------|---------------|---------------|
    | Игрок: 0 (C) | (3, 3)       | (0, 5)       |
    | Игрок: 1 (D) | (5, 0)       | (1, 1)       |
    """
    payoff_matrix = {
        (0, 0): (3, 3),  # Оба сотрудничают
        (0, 1): (0, 5),  # Игрок 1 сотрудничает, Игрок 2 предает
        (1, 0): (5, 0),  # Игрок 1 предает, Игрок 2 сотрудничает
        (1, 1): (1, 1)  # Оба предают
    }
    return payoff_matrix[(decision1, decision2)]


def simulate_prisoners_dilemma(
        strategy1_name: str,
        strategy2_name: str,
        rounds: int = 10,
        verbose: bool = False,
        noise_prob: float = 0.02
) -> Tuple[int, int, List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Симулирует игру в дилемму заключенного между двумя агентами с заданными стратегиями

    Args:
        strategy1_name: Имя стратегии первого агента (из STRATEGIES)
        strategy2_name: Имя стратегии второго агента (из STRATEGIES)
        rounds: Количество раундов игры

    Returns:
        Tuple[int, int, List[Tuple[int, int]], List[Tuple[int, int]]]:
        - total_payoff1: Общий выигрыш первого агента
        - total_payoff2: Общий выигрыш второго агента
        - history: История взаимодействий в формате [(decision_agent1, decision_agent2), ...]
        - payoffs_history: История выигрышей за каждый раунд [(payoff1, payoff2), ...]

    Raises:
        ValueError: Если указано несуществующее имя стратегии
    """
    # Проверка существования стратегий
    if strategy1_name not in STRATEGIES:
        raise ValueError(f"Неизвестная стратегия для первого агента: {strategy1_name}")
    if strategy2_name not in STRATEGIES:
        raise ValueError(f" Неизвестная стратегия для второго агента: {strategy2_name}")

    # Получение функций стратегий по именам
    strategy1 = STRATEGIES[strategy1_name]
    strategy2 = STRATEGIES[strategy2_name]

    # Позиции агентов: 0 - первый агент, 1 - второй агент
    agent1_position = 0
    agent2_position = 1

    total_payoff1 = 0
    total_payoff2 = 0
    history = []
    payoffs_history = []
    if verbose:
        print(f"🎮 Запуск симуляции: {strategy1_name} vs {strategy2_name} ({rounds} раундов)")
        print("-" * 60)

    for round_num in range(rounds):
        # Получаем решения агентов
        decision1 = strategy1(history.copy(), agent1_position, agent2_position)
        decision2 = strategy2(history.copy(), agent2_position, agent1_position)
        if random.random() < noise_prob:
            decision1 = 1 - decision1  # Инвертируем решение агента 1
        if random.random() < noise_prob:
            decision2 = 1 - decision2  # Инвертируем решение агента 2
        # Рассчитываем выигрыши за раунд
        payoff1, payoff2 = calculate_payoffs(decision1, decision2)

        # Обновляем общие выигрыши
        total_payoff1 += payoff1
        total_payoff2 += payoff2

        # Добавляем в историю
        history.append((decision1, decision2))
        payoffs_history.append((payoff1, payoff2))

        if verbose:
            # Отладочный вывод
            print(f" Раунд {round_num + 1}:")
            print(f"   Агент 1 ({strategy1_name}): {'Сотрудничает' if decision1 == 0 else 'Предает'}")
            print(f"   Агент 2 ({strategy2_name}): {'Сотрудничает' if decision2 == 0 else 'Предает'}")
            print(f"   Выигрыши: Агент 1 = {payoff1}, Агент 2 = {payoff2}")
            print(f"   Общие выигрыши: Агент 1 = {total_payoff1}, Агент 2 = {total_payoff2}")
            print("-" * 40)
    if verbose:
        print(f"🎯 ИТОГОВЫЕ РЕЗУЛЬТАТЫ:")
        print(f"   Агент 1 ({strategy1_name}): {total_payoff1} очков")
        print(f"   Агент 2 ({strategy2_name}): {total_payoff2} очков")
        print(f"   История взаимодействий: {history}")
        print("=" * 60)

    return total_payoff1, total_payoff2, history, payoffs_history

GAME_RULES = """РЕГЛАМЕНТ ПОВТОРЯЮЩЕЙСЯ ДИЛЕММЫ ЗАКЛЮЧЕННЫХ

🎯 СУТЬ ИГРЫ:
Вы участвуете в многораундовой игре с другим агентом. В каждом раунде вы независимо выбираете одно из двух действий:
- 0 (СОТРУДНИЧАТЬ): работать вместе с партнером
- 1 (ПРЕДАТЬ): действовать исключительно в своих интересах

📊 МАТРИЦА ВЫИГРЫШЕЙ (очки за один раунд):
┌─────────────────┬─────────────────┬─────────────────┐
│ Ваше действие   │ Действие партнера │ Ваши очки       │
├─────────────────┼─────────────────┼─────────────────┤
│ Сотрудничать (0)│ Сотрудничать (0)│ +3              │
│ Сотрудничать (0)│ Предать (1)     │ +0              │
│ Предать (1)     │ Сотрудничать (0)│ +5              │
│ Предать (1)     │ Предать (1)     │ +1              │
└─────────────────┴─────────────────┴─────────────────┘

⚖️ ВАЖНО:
- Игра состоит из множества раундов (точное количество неизвестно
- Каждое взаимодействие записывается в историю
- Ваша репутация влияет на будущие взаимодействия с другими агентами
- Оптимальная стратегия зависит от стратегии вашего партнера
"""

