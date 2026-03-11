from spade_llm.demo.platform.prisoners_dillema.agents import *
import seaborn as sns
import pandas as pd
import numpy as np
import os
import math
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from collections import defaultdict


def a2a_visualisation():
    """
    Генерация графиков динамики стратегий для каждого агента:
    Все взаимодействия агента с разными оппонентами отображаются на одном графике
    с использованием подграфиков для каждого оппонента
    """
    os.makedirs('results', exist_ok=True)

    # Фильтрация только LLM-агентов (без фиксированных стратегий)
    LLM_AGENTS = [
        aid for aid in AGENT_STATS.keys()
        if get_fixed_strategy(aid, TOPOLOGY) is None
           or COMMUNITY_STRATEGY.get(TOPOLOGY.community_of.get(aid)) is None
    ]

    for agent_id in LLM_AGENTS:
        if agent_id not in DETAILED_STRATEGY_HISTORY:
            continue

        opponents = DETAILED_STRATEGY_HISTORY[agent_id]
        valid_opponents = {oid: data for oid, data in opponents.items() if len(data) >= 2}

        if not valid_opponents:
            continue

        # Определяем количество подграфиков (максимум 9 для читаемости)
        n_opponents = len(valid_opponents)
        max_plots = min(n_opponents, 9)  # Максимум 9 подграфиков на одном изображении

        if max_plots == 0:
            continue

        # Определяем сетку подграфиков (3x3 максимум)
        n_cols = min(3, max_plots)
        n_rows = math.ceil(max_plots / n_cols)

        # Создаем фигуру с подграфиками
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows),
                                 sharex=True, sharey=True)

        # Преобразуем axes в двумерный массив для удобства
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1 or n_cols == 1:
            axes = np.array(axes).reshape(n_rows, n_cols)

        # Список всех стратегий для единой оси Y
        all_strategies = set()
        for interactions in valid_opponents.values():
            strategies = [s for _, s in interactions]
            all_strategies.update(strategies)

        unique_strategies = sorted(all_strategies)
        y_pos = range(len(unique_strategies))
        strategy_to_y = {s: i for i, s in enumerate(unique_strategies)}

        # Строим графики для каждого оппонента
        for idx, (opponent_id, interactions) in enumerate(valid_opponents.items()):
            if idx >= max_plots:  # Ограничиваем количество подграфиков
                break

            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            # Сортируем по номеру раунда
            interactions_sorted = sorted(interactions, key=lambda x: x[0])
            round_nums = [r for r, s in interactions_sorted]
            strategies = [s for r, s in interactions_sorted]

            # Преобразуем стратегии в числовые значения
            y_values = [strategy_to_y[s] for s in strategies]
            colors = [STRATEGY_COLORS.get(s, 'black') for s in strategies]

            # Строим scatter plot
            scatter = ax.scatter(
                round_nums,
                y_values,
                s=100,
                c=colors,
                edgecolors='black',
                alpha=0.9,
                zorder=3
            )

            # Добавляем линии для визуализации переходов
            if len(round_nums) > 1:
                ax.plot(round_nums, y_values,
                        color='gray', alpha=0.4, linestyle='--', linewidth=1)

            # Подписываем каждую точку
            for i, (r, s) in enumerate(zip(round_nums, strategies)):
                ax.text(
                    r,
                    y_values[i] + 0.15,
                    s,
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1)
                )

            # Настройка подграфика
            ax.set_title(f'Агент {agent_id} vs {opponent_id}', fontsize=12, pad=10)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(unique_strategies)
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            ax.set_ylim(-0.5, len(unique_strategies) - 0.5)

            # Добавляем вертикальные линии для каждого раунда
            for r in set(round_nums):
                ax.axvline(x=r, color='gray', alpha=0.2, linestyle=':')

        # Удаляем пустые подграфики
        total_plots = n_rows * n_cols
        used_plots = min(max_plots, total_plots)
        for idx in range(used_plots, total_plots):
            row = idx // n_cols
            col = idx % n_cols
            fig.delaxes(axes[row, col])

        # Настройка общей оси X
        for col in range(n_cols):
            axes[-1, col].set_xlabel('Номер раунда', fontsize=10)

        # Настройка общей оси Y для первого столбца
        for row in range(n_rows):
            axes[row, 0].set_ylabel('Стратегия', fontsize=10)

        # Общий заголовок
        fig.suptitle(f'Динамика выбора стратегий агента {agent_id}',
                     fontsize=16, fontweight='bold', y=0.98)

        # Легенда стратегий (общая для всей фигуры)
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=STRATEGY_COLORS.get(s, 'black'),
                   markersize=10, label=s)
            for s in unique_strategies
        ]

        # Размещаем легенду вне графика
        fig.legend(
            handles=legend_elements,
            title="Стратегии",
            loc='upper center',
            bbox_to_anchor=(0.5, 0.02),
            ncol=min(len(unique_strategies), 5),
            frameon=True,
            shadow=True,
            fontsize=9
        )

        # Автоматическая подгонка
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Оставляем место для легенды внизу

        # Сохранение
        filename = f"results/agent_{agent_id}_all_interactions.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Сохранен график всех взаимодействий агента {agent_id} ({len(valid_opponents)} оппонентов)")

def community_layout(G, communities, distance_factor=1.5):
    """
    Укладка графа с учетом сообществ, чтобы сообщества не перекрывались.

    Parameters:
    - G: граф с рёбрами
    - communities: словарь сообществ
    - distance_factor: коэффициент для увеличения расстояния между сообществами
    """
    pos = {}
    community_centers = {}

    # Определяем центр для каждого сообщества
    for community_id, agents in communities.items():
        community_center = (random.uniform(-10, 10), random.uniform(-10, 10))
        community_centers[community_id] = community_center

    # Для каждого агента позиционируем его относительно его сообщества
    for community_id, agents in communities.items():
        center = community_centers[community_id]
        for i, agent in enumerate(agents):
            # Размещаем агента вокруг центра сообщества с учетом случайности
            angle = 2 * math.pi * i / len(agents)
            pos[agent] = (
                center[0] + distance_factor * random.uniform(0.8, 1.2) * math.cos(angle),
                center[1] + distance_factor * random.uniform(0.8, 1.2) * math.sin(angle)
            )

    return pos


# Ваша функция визуализации
def generate_community_layout_graph(G, communities, ROUND_STATS, AGENT_STATS):
    pos = community_layout(G, communities)

    # Подготовка визуальных параметров
    weights = [d["weight"] for _, _, d in G.edges(data=True)]
    norm = Normalize(vmin=min(weights), vmax=max(weights))
    cmap = cm.get_cmap("RdYlGn")
    edge_colors = [cmap(norm(w)) for w in weights]
    edge_widths = [1.5 + 3.0 * norm(w) for w in weights]

    # Визуализация графа
    fig, ax = plt.subplots(figsize=(12, 12))
    nx.draw_networkx_nodes(
        G, pos,
        node_size=1200,
        node_color="lightgray",
        edgecolors="black",
        ax=ax
    )
    nx.draw_networkx_labels(
        G, pos,
        font_size=12,
        font_weight="bold",
        ax=ax
    )
    nx.draw_networkx_edges(
        G, pos,
        arrowstyle="->",
        arrowsize=20,
        edge_color=edge_colors,
        width=edge_widths,
        connectionstyle="arc3,rad=0.15",
        ax=ax
    )

    # Добавление цветовой шкалы
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax, shrink=0.75)
    cbar.set_label("Средний выигрыш агента i против j")

    ax.set_title(
        "Ориентированный полносвязный граф средних выигрышей\n"
        "Ребро i → j = средний выигрыш агента i при игре с j",
        fontsize=14
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig("results/directed_avg_payoff_graph_with_community_layout.png", dpi=300)
    plt.close()

def generate_directed_payoff_graph():
    """
    Ориентированный полносвязный граф:
    i → j — средний выигрыш агента i при взаимодействии с агентом j
    """


    os.makedirs("results", exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Агрегация выигрышей i -> j
    # ------------------------------------------------------------------
    payoff_sum = defaultdict(float)
    payoff_count = defaultdict(int)

    for round_data in ROUND_STATS:
        for inter in round_data["interactions"]:
            i, j = map(int, inter["pair"].split("-"))

            payoff_sum[(i, j)] += inter["agent1_score"]
            payoff_sum[(j, i)] += inter["agent2_score"]

            payoff_count[(i, j)] += 1
            payoff_count[(j, i)] += 1

    avg_payoff = {
        k: payoff_sum[k] / payoff_count[k]
        for k in payoff_sum
        if payoff_count[k] > 0
    }

    # ------------------------------------------------------------------
    # 2. Построение ориентированного полносвязного графа
    # ------------------------------------------------------------------
    G = nx.DiGraph()
    agents = sorted(AGENT_STATS.keys())
    G.add_nodes_from(agents)

    for (i, j), value in avg_payoff.items():
        G.add_edge(i, j, weight=value)

    # ------------------------------------------------------------------
    # 3. Подготовка визуальных параметров
    # ------------------------------------------------------------------
    weights = [d["weight"] for _, _, d in G.edges(data=True)]

    norm = Normalize(vmin=min(weights), vmax=max(weights))
    cmap = cm.get_cmap("RdYlGn")

    edge_colors = [cmap(norm(w)) for w in weights]
    edge_widths = [1.5 + 3.0 * norm(w) for w in weights]

    # ------------------------------------------------------------------
    # 4. Визуализация (КЛЮЧЕВО: явный Figure / Axes)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 12))

    pos = nx.circular_layout(G)

    nx.draw_networkx_nodes(
        G, pos,
        node_size=1200,
        node_color="lightgray",
        edgecolors="black",
        ax=ax
    )

    nx.draw_networkx_labels(
        G, pos,
        font_size=12,
        font_weight="bold",
        ax=ax
    )

    nx.draw_networkx_edges(
        G, pos,
        arrowstyle="->",
        arrowsize=20,
        edge_color=edge_colors,
        width=edge_widths,
        connectionstyle="arc3,rad=0.15",
        ax=ax
    )

    # ------------------------------------------------------------------
    # 5. Корректная цветовая шкала
    # ------------------------------------------------------------------
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax, shrink=0.75)
    cbar.set_label("Средний выигрыш агента i против j")

    ax.set_title(
        "Ориентированный полносвязный граф средних выигрышей\n"
        "Ребро i → j = средний выигрыш агента i при игре с j",
        fontsize=14
    )

    ax.axis("off")
    plt.tight_layout()
    plt.savefig("results/directed_avg_payoff_graph.png", dpi=300)
    plt.close()
    # ------------------------------------------------------------------
    generate_community_layout_graph(
        G,  # Граф с выигрышами
        COMMUNITIES,  # Сообщества
        ROUND_STATS,  # Статистика раундов
        AGENT_STATS   # Статистика агентов
    )

STRATEGY_COLORS = {
    "AC": "#2ca02c",    # зелёный — кооперация
    "TFT": "#1f77b4",   # синий — условная кооперация
    "FTFT": "#17becf",  # голубой — прощение
    "GT": "#9467bd",    # фиолетовый — жёсткая условность
    "AD": "#d62728",    # красный — дефекция
    "RAND": "#7f7f7f"   # серый — хаос
}
def get_dominant_strategy(agent_stats: dict) -> str:
    if not agent_stats["strategy_counts"]:
        return "RAND"
    return max(
        agent_stats["strategy_counts"].items(),
        key=lambda x: x[1]
    )[0]

def generate_visualizations():
    """Генерация визуализаций по собранным данным"""
    os.makedirs('results', exist_ok=True)

    # 1. Общие выигрыши агентов
    plt.figure(figsize=(12, 6))
    agents = sorted(AGENT_STATS.keys())
    mean_scores = [
        AGENT_STATS[a]['total_score'] / AGENT_STATS[a]['n_interactions']
        for a in agents
    ]
    dominant_strategies = [
        get_dominant_strategy(AGENT_STATS[a])
        for a in agents
    ]
    bar_colors = [
        STRATEGY_COLORS.get(strategy, "#000000")
        for strategy in dominant_strategies
    ]
    bars = plt.bar(agents, mean_scores, color=bar_colors)
    plt.title('Средний выигрыш агента на одно взаимодействие')
    plt.xlabel('ID агента')
    plt.ylabel('Mean score')
    plt.xticks(agents)
    # Подписи значений
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    # Легенда стратегий
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color, label=strategy)
        for strategy, color in STRATEGY_COLORS.items()
    ]
    plt.legend(
        handles=legend_elements,
        title="Превалирующая стратегия",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )
    plt.tight_layout()
    plt.savefig("results/mean_score_by_agent_and_strategy.png", dpi=300)
    plt.close()

    # 2. Распределение стратегий
    plt.figure(figsize=(15, 8))
    strategy_data = []

    for agent_id, stats in AGENT_STATS.items():
        for strategy, count in stats['strategy_counts'].items():
            strategy_data.append({
                'agent': f"Агент {agent_id}",
                'strategy': strategy,
                'count': count
            })

    df_strategies = pd.DataFrame(strategy_data)
    sns.barplot(data=df_strategies, x='agent', y='count', hue='strategy')
    plt.title('Распределение стратегий по агентам')
    plt.xlabel('Агент')
    plt.ylabel('Количество выборов')
    plt.xticks(rotation=45)
    plt.legend(title='Стратегия')
    plt.tight_layout()
    plt.savefig('results/strategy_distribution.png')
    plt.close()

    # 3. Динамика выигрышей по раундам
    plt.figure(figsize=(12, 6))

    for agent_id in sorted(AGENT_STATS.keys()):
        round_scores = []
        cumulative = 0
        for round_data in ROUND_STATS:
            for interaction in round_data['interactions']:
                if interaction['pair'].startswith(f"{agent_id}-") or interaction['pair'].endswith(f"-{agent_id}"):
                    if interaction['pair'].startswith(f"{agent_id}-"):
                        cumulative += interaction['agent1_score']
                    else:
                        cumulative += interaction['agent2_score']
            round_scores.append(cumulative)

        plt.plot(range(1, len(ROUND_STATS) + 1), round_scores,
                 label=f"Агент {agent_id}",
                 marker='o',
                 linewidth=2)

    plt.title('Динамика накопления очков по раундам')
    plt.xlabel('Номер раунда')
    plt.ylabel('Накопленные очки')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/score_dynamics.png')
    plt.close()
    generate_directed_payoff_graph()
    a2a_visualisation()
    logger.info("Визуализации сохранены в папку results/")
