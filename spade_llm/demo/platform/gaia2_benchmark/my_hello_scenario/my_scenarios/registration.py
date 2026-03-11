def register_scenarios(registry):
    """
    Регистрируем все сценарии в пакете.
    """
    # Импорт модуля активирует декоратор @register_scenario
    import my_scenarios.hello_scenario