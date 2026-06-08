import logging
import math
import random

logger = logging.getLogger(__name__)


def logit_probability_of_purchase(price: float, max_price: float, sensitivity: float = 0.05) -> float:
    """
    Вероятность, что пользователь купит товар по данной цене.
    Чем выше sensitivity, тем резче спад вероятности после max_price.
    """
    prob = 1 / (1 + math.exp(sensitivity * (price - max_price)))
    logger.info("Верояность покупки: %s", prob)
    return prob


def shifted_left_exp_purchase_prob(price: float, max_price: float, shift: float = 0.0, decay: float = 0.08) -> float:
    """
    Вероятность покупки с экспоненциальным спадом, начинающимся раньше лимита.

    Параметры:
    ----------
    price : float
        Текущая цена набора.
    max_price : float
        Лимит — цена, которую пользователь считает комфортной.
    shift : float
        Насколько раньше начинается спад вероятности (в тех же единицах, что и цена).
        Например, shift=20 означает, что сомнение начинается при цене max_price - 20.
    decay : float
        Скорость экспоненциального спада после начала снижения вероятности.
        Большее значение = резче падение вероятности.

    Возвращает:
    -----------
    Вероятность покупки от 0 до 1.
    """

    # Точка, где вероятность начинает падать
    start_price = max_price - shift

    if price <= start_price:
        prob = 1.0
    else:
        prob = math.exp(-decay * (price - start_price))
    logger.info("Верояность покупки: %s", prob)
    return prob


def user_decision(price: float, max_price: float, function: str = "logit", shift: float = 0.0, decay: float = 0.05,
                  sensitivity: float = 0.05) -> bool:
    """
    Симуляция факта покупки с вероятностью на основе логистической функции.
    """
    if function == "logit":
        p = logit_probability_of_purchase(price, max_price, sensitivity)
    elif function == "exp":
        p = shifted_left_exp_purchase_prob(price, max_price, shift, decay)
    else:
        raise ValueError(f"Неизвестная функция: {function}")
    result = random.random() < p
    logger.info("Покупатель решил купить: %s", result)
    return result
