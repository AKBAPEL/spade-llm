import json
import logging
import math
import os
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

RUN_ID_ENV = "AUCTION_RUN_ID"
MECHANISM_ENV = "AUCTION_MECHANISM"


def get_artifacts_dir() -> Path:
    """Return the central artifacts directory for the auction package."""
    return Path(__file__).resolve().parent / "artifacts"


def ensure_artifact_dirs() -> Dict[str, Path]:
    """Create and return the standard artifact subdirectories."""
    base = get_artifacts_dir()
    dirs = {
        "dialogues": base / "dialogues",
        "auction_logs": base / "auction_logs",
        "memory": base / "data" / "memory",
        "metadata": base / "metadata",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def get_run_id() -> str:
    """Return the current run id, creating one from timestamp if not set."""
    run_id = os.environ.get(RUN_ID_ENV)
    if not run_id:
        run_id = datetime.now().strftime("%d-%H-%M-%S-%f")
        os.environ[RUN_ID_ENV] = run_id
    return run_id


def get_mechanism(default: str = "unknown") -> str:
    """Return the trust mechanism label for the current run."""
    return os.environ.get(MECHANISM_ENV, default)


def save_run_meta(extra: Optional[Dict[str, Any]] = None) -> Path:
    """Persist run metadata so reports can be grouped by mechanism/run."""
    meta = {
        "run_id": get_run_id(),
        "mechanism": get_mechanism(),
        "timestamp": datetime.now().isoformat(),
    }
    if extra:
        meta.update(extra)
    dirs = ensure_artifact_dirs()
    path = dirs["metadata"] / f"run_meta_{meta['run_id']}.json"
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved run metadata to %s", path)
    return path


def archive_current_run(output_dir: Path, label: str) -> Path:
    """Move all artifacts into backup_artifacts/<label>_<run_id> and clean artifacts dir."""
    run_id = get_run_id()
    artifacts = get_artifacts_dir()
    backup_root = artifacts.parent / "backup_artifacts"
    backup_dir = backup_root / f"{label}_{run_id}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    moved = 0
    for sub in ["dialogues", "auction_logs", "data/memory", "metadata"]:
        src = artifacts / sub
        if not src.exists():
            continue
        dst = backup_dir / sub
        dst.mkdir(parents=True, exist_ok=True)
        for item in src.iterdir():
            target = dst / item.name
            if item.is_file():
                shutil.move(str(item), str(target))
                moved += 1
            elif item.is_dir():
                shutil.move(str(item), str(target))
                moved += 1

    logger.info("Archived %d artifact items from %s to %s", moved, artifacts, backup_dir)
    return backup_dir


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
