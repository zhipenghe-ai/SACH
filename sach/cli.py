"""Shared implementation for the three dataset-specific launchers."""

from pathlib import Path
import warnings

from .config import build_parser
from .run_logging import logger


def log_config(log, config):
    log.info(">>> Configs List <<<")
    fields = (
        ("Dadaset", "DATASET"),
        ("SEED", "SEED"),
        ("Bit", "HASH_BIT"),
        ("Batch", "BATCH_SIZE"),
        ("Lr_IMG", "LR_IMG"),
        ("Lr_TXT", "LR_TXT"),
        ("LR_MyNet", "LR_MyNet"),
        ("lambda1", "lambda1"),
        ("lambda2", "lambda2"),
        ("beta", "beta"),
        ("K", "K"),
        ("a1", "a1"),
        ("a2", "a2"),
    )
    for label, attribute in fields:
        log.info(f"--- {label}:{getattr(config, attribute)}")


def run(config, preset):
    if not config.TRAIN:
        from .evaluate import run_evaluation

        return run_evaluation(config)

    import torch

    from .trainer import PCIRH

    warnings.filterwarnings("ignore")
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    torch.cuda.set_device(config.GPU_ID)

    log_dir = Path(getattr(config, "LOG_DIR", "./logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log = logger(f"{config.DATASET}_{config.HASH_BIT}", log_dir=log_dir)
    log_config(log, config)

    trainer = PCIRH(log, config)
    best_it = best_ti = 0
    for epoch in range(config.NUM_EPOCH):
        codes, indices = trainer.train_method(epoch)
        trainer.train_Hashfunc(codes, indices, epoch)

        if (epoch + 1) % config.EVAL_INTERVAL == 0:
            map_it, map_ti, top_it, top_ti, p_it, r_it, p_ti, r_ti = trainer.performance_eval()
            log.info(
                "MAP (full gallery) I->T: %.3f, MAP (full gallery) T->I: %.3f" % (map_it, map_ti)
            )
            if preset != "COCO":
                log.info("I->T: %s, Shape: %s", top_it, top_it.shape)
                log.info("T->I: %s, Shape: %s", top_ti, top_ti.shape)
                log.info("I->T p: %s, Shape: %s", p_it, p_it.shape)
                log.info("T->I r: %s, Shape: %s", r_it, r_it.shape)
                log.info("I->T p: %s, Shape: %s", p_ti, p_ti.shape)
                log.info("T->I r: %s, Shape: %s", r_ti, r_ti.shape)

            if (best_it + best_ti) < (map_it + map_ti):
                best_it, best_ti = map_it, map_ti
                log.info(
                    "Best MAP (full gallery) of I->T: %.3f, Best MAP (full gallery) of T->I: %.3f"
                    % (best_it, best_ti)
                )
                trainer.save_checkpoints()

            log.info("--------------------------------------------------------------------")


def main(preset, argv=None):
    parser = build_parser(preset)
    config = parser.parse_args(argv)
    try:
        return run(config, preset)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        if config.TRAIN:
            raise
        parser.error(str(exc))
