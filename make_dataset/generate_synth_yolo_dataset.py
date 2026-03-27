import os
import sys
import csv
import random
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

import numpy as np

# ============================================================
# Генератор синтетического датасета (YOLO)
# ============================================================
# В этом файле оставляем ОСНОВНУЮ оркестрацию (`generate_dataset`),
# а вспомогательные функции (загрузка/эффекты/геометрия/IO) лежат в:
# - make_dataset/config.py
# - make_dataset/effects.py
# - make_dataset/utils.py

if __name__ == "__main__" and __package__ is None:
    # Чтобы корректно работали импорты "from make_dataset import ..."
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from make_dataset import config as cfg  # noqa: E402
from make_dataset import effects  # noqa: E402
from make_dataset.utils import (  # noqa: E402
    HAS_CAIROSVG,
    build_template_sequence,
    compute_target_count,
    ensure_output_dir,
    generate_multi,
    import_external_images_for_selected,
    info,
    load_background_paths,
    load_bg_bgr,
    load_classes,
    load_external_index,
    normalize_class_code,
    reserve_object_specs,
    save_negative_sample,
    save_sample,
    warn,
    worker_generate_and_save,
    write_classes_txt,
    write_dataset_yaml_and_splits,
)

__all__ = ["generate_dataset"]


def generate_dataset():
    random.seed(cfg.RANDOM_SEED)
    np.random.seed(cfg.RANDOM_SEED)

    bg_paths = load_background_paths()

    all_classes = load_classes(filter_ids=None)

    if cfg.SELECT_CLASS_IDS:
        selected_ids = sorted(set(int(x) for x in cfg.SELECT_CLASS_IDS))
    else:
        selected_ids = [int(c["class_id"]) for c in all_classes]

    selected_id_set = set(selected_ids)
    selected_classes = [c for c in all_classes if int(c["class_id"]) in selected_id_set]
    if not selected_classes:
        raise RuntimeError("Нет выбранных классов для генерации (проверьте SELECT_CLASS_IDS).")

    our_code_to_id = {normalize_class_code(c["name"]): int(c["class_id"]) for c in all_classes}

    external_index = {"enabled": False, "instances_per_class": {}, "images_by_class": {}, "images": {}}
    if cfg.EXTERNAL_MIX_ENABLED and os.path.isdir(cfg.EXTERNAL_DATASET_DIR):
        external_index = load_external_index(cfg.EXTERNAL_DATASET_DIR, our_code_to_id)

    target_instances = compute_target_count(selected_classes, external_index.get("instances_per_class"))
    info(
        f"Классов всего: {len(all_classes)}. Выбрано для генерации: {len(selected_classes)}. "
        f"TARGET (экземпляров на выбранный класс) = {target_instances}."
    )

    if not HAS_CAIROSVG:
        warn("cairosvg не установлен. Будем пробовать rsvg-convert/inkscape для SVG.")

    out_dir = ensure_output_dir(cfg.OUTPUT_BASE)
    info(f"Выходная папка: {out_dir}")

    write_classes_txt(out_dir, all_classes)

    train_list = []
    val_list = []

    ann_csv_path = os.path.join(out_dir, "annotations.csv")
    ann_f = open(ann_csv_path, "w", encoding="utf-8", newline="")
    ann_w = csv.writer(ann_f)
    ann_w.writerow(["image_file", "class_id", "x_center", "y_center", "width", "height"])

    template_cache = {}
    unique_idx = 0

    id_to_code = {int(c["class_id"]): str(c["name"]) for c in all_classes}

    state_by_id = {}
    for c in all_classes:
        cid = int(c["class_id"])
        if cid in selected_id_set:
            seq = build_template_sequence(c, target_instances)
            state_by_id[cid] = {"cls": c, "seq": seq, "ptr": 0, "count": 0, "target": target_instances}
        else:
            state_by_id[cid] = {
                "cls": c,
                "seq": [random.choice(c["templates"])],
                "ptr": 0,
                "count": 0,
                "target": 0,
            }

    states = list(state_by_id.values())

    if external_index.get("enabled"):
        unique_idx_ref = [unique_idx]
        imported_counts, _imported_ids = import_external_images_for_selected(
            out_dir=out_dir,
            external_index=external_index,
            selected_ids=selected_ids,
            id_to_code=id_to_code,
            train_list=train_list,
            val_list=val_list,
            ann_w=ann_w,
            unique_idx_ref=unique_idx_ref,
        )
        unique_idx = unique_idx_ref[0]
        for cid, cnt in imported_counts.items():
            if cid in state_by_id:
                state_by_id[cid]["count"] += int(cnt)
                if state_by_id[cid]["target"] < state_by_id[cid]["count"]:
                    state_by_id[cid]["target"] = state_by_id[cid]["count"]

    try:
        selected_states = [state_by_id[cid] for cid in selected_id_set if cid in state_by_id]
        total_target = len(selected_states) * target_instances
        total_made = sum(s["count"] for s in selected_states)
        attempts = 0
        max_attempts = total_target * cfg.MAX_ATTEMPTS_MULT

        if cfg.SELECT_CLASS_IDS and not cfg.ALLOW_EXTRA_NON_SELECTED_CLASSES:
            extra_pool_states = selected_states
        else:
            extra_pool_states = states

        if cfg.USE_MULTITHREADING and cfg.NUM_WORKERS > 1:
            max_inflight = cfg.MAX_INFLIGHT_TASKS if cfg.MAX_INFLIGHT_TASKS is not None else cfg.NUM_WORKERS
            inflight = {}

            with ThreadPoolExecutor(max_workers=cfg.NUM_WORKERS) as ex:
                while total_made < total_target and attempts < max_attempts:
                    remaining = max(0, total_target - total_made)
                    cur_max_inflight = max(1, min(max_inflight, remaining))

                    while len(inflight) < cur_max_inflight and attempts < max_attempts:
                        need = [s for s in selected_states if s["count"] < s["target"]]
                        if not need:
                            break

                        weights = [max(1, s["target"] - s["count"]) for s in need]
                        main_state = random.choices(need, weights=weights, k=1)[0]

                        specs = reserve_object_specs(main_state, extra_pool_states)
                        if not specs:
                            attempts += 1
                            continue

                        unique_idx += 1
                        fut = ex.submit(worker_generate_and_save, out_dir, bg_paths, specs, unique_idx, id_to_code)
                        inflight[fut] = True
                        attempts += 1

                    if not inflight:
                        break

                    done, _pending = wait(inflight.keys(), return_when=FIRST_COMPLETED)
                    for fut in done:
                        inflight.pop(fut, None)
                        try:
                            res = fut.result()
                        except Exception as e:
                            warn(f"Ошибка в worker: {e}")
                            continue

                        if res is None:
                            continue

                        rel_img_path, labels = res
                        if random.random() < cfg.VAL_RATIO:
                            val_list.append(rel_img_path)
                        else:
                            train_list.append(rel_img_path)

                        for (cid, xc, yc, ww, hh) in labels:
                            ann_w.writerow([rel_img_path, cid, f"{xc:.6f}", f"{yc:.6f}", f"{ww:.6f}", f"{hh:.6f}"])
                            if int(cid) in state_by_id:
                                state_by_id[int(cid)]["count"] += 1

                        total_made = sum(s["count"] for s in selected_states)
        else:
            while total_made < total_target and attempts < max_attempts:
                attempts += 1
                need = [s for s in selected_states if s["count"] < s["target"]]
                if not need:
                    break

                weights = [max(1, s["target"] - s["count"]) for s in need]
                main_state = random.choices(need, weights=weights, k=1)[0]

                res = generate_multi(bg_paths, main_state, states, extra_pool_states, template_cache)
                if res is None:
                    continue

                img_bgr, labels = res
                img_bgr = effects.apply_weather(img_bgr)
                img_bgr = effects.apply_camera_effects(img_bgr)

                unique_idx += 1
                rel_img_path = save_sample(out_dir, img_bgr, unique_idx, labels, id_to_code)

                if random.random() < cfg.VAL_RATIO:
                    val_list.append(rel_img_path)
                else:
                    train_list.append(rel_img_path)

                for (cid, xc, yc, ww, hh) in labels:
                    ann_w.writerow([rel_img_path, cid, f"{xc:.6f}", f"{yc:.6f}", f"{ww:.6f}", f"{hh:.6f}"])
                    if int(cid) in state_by_id:
                        state_by_id[int(cid)]["count"] += 1

                total_made = sum(s["count"] for s in selected_states)

        for s in selected_states:
            c = s["cls"]
            if s["count"] < s["target"]:
                warn(f"Класс id={c['class_id']} набрал {s['count']}/{s['target']} экземпляров.")
            else:
                info(f"Класс id={c['class_id']}: {s['count']}/{s['target']} экземпляров.")
    finally:
        ann_f.close()

    exp_objs = 1.0 + (cfg.EXTRA_OBJECTS_RANGE[0] + cfg.EXTRA_OBJECTS_RANGE[1]) / 2.0 if cfg.MULTI_OBJECT_ENABLED else 1.0
    exp_images = (len(selected_classes) * target_instances) / max(1.0, exp_objs)
    neg_count = int(round(exp_images * float(cfg.NEGATIVE_RATIO)))
    if neg_count > 0:
        info(f"Генерируем негативные изображения: {neg_count}")
        for _ in range(neg_count):
            bg = load_bg_bgr(random.choice(bg_paths))
            bg = effects.apply_weather(bg)
            bg = effects.apply_camera_effects(bg)
            unique_idx += 1
            rel_img_path = save_negative_sample(out_dir, bg, unique_idx)
            if random.random() < cfg.VAL_RATIO:
                val_list.append(rel_img_path)
            else:
                train_list.append(rel_img_path)

    write_dataset_yaml_and_splits(out_dir, all_classes, train_list, val_list)
    info("Готово.")
    return out_dir


if __name__ == "__main__":
    generate_dataset()
