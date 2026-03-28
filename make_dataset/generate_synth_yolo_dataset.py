import os
import sys
import csv
import random
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from collections import OrderedDict

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

from make_dataset import config as cfg
from make_dataset import effects
from make_dataset.utils import (
    HAS_CAIROSVG,
    build_template_sequence,
    compute_target_count,
    count_dir_files,
    ensure_output_dir,
    fmt_progress,
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
    write_data_stats,
    write_dataset_yaml_and_splits,
)


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

    stream_splits = bool(getattr(cfg, "STREAM_SPLITS_TO_DISK", False))
    train_list = None if stream_splits else []
    val_list = None if stream_splits else []
    train_f = None
    val_f = None
    if stream_splits:
        train_f = open(os.path.join(out_dir, "train.txt"), "w", encoding="utf-8")
        val_f = open(os.path.join(out_dir, "val.txt"), "w", encoding="utf-8")

    ann_csv_path = os.path.join(out_dir, "annotations.csv")
    ann_f = open(ann_csv_path, "w", encoding="utf-8", newline="")
    ann_w = csv.writer(ann_f)
    ann_w.writerow(["image_file", "class_id", "x_center", "y_center", "width", "height"])

    template_cache = OrderedDict()
    unique_idx = 0

    id_to_code = {int(c["class_id"]): str(c["name"]) for c in all_classes}

    imported_counts = {}
    generated_counts = {}

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
            train_f=train_f,
            val_f=val_f,
        )
        unique_idx = unique_idx_ref[0]
        for cid, cnt in imported_counts.items():
            if cid in state_by_id:
                state_by_id[cid]["count"] += int(cnt)
                if state_by_id[cid]["target"] < state_by_id[cid]["count"]:
                    state_by_id[cid]["target"] = state_by_id[cid]["count"]

    try:
        info("Начало генерации изображений")
        selected_states = [state_by_id[cid] for cid in selected_id_set if cid in state_by_id]
        total_target = len(selected_states) * target_instances
        total_made = sum(int(imported_counts.get(int(cid), 0)) for cid in selected_id_set)
        attempts = 0
        max_attempts = total_target * cfg.MAX_ATTEMPTS_MULT
        last_report_at = -1

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
                            if val_list is not None:
                                val_list.append(rel_img_path)
                            elif val_f is not None:
                                val_f.write(rel_img_path + "\n")
                        else:
                            if train_list is not None:
                                train_list.append(rel_img_path)
                            elif train_f is not None:
                                train_f.write(rel_img_path + "\n")

                        for (cid, xc, yc, ww, hh) in labels:
                            cid_i = int(cid)
                            ann_w.writerow([rel_img_path, cid, f"{xc:.6f}", f"{yc:.6f}", f"{ww:.6f}", f"{hh:.6f}"])
                            generated_counts[cid_i] = generated_counts.get(cid_i, 0) + 1
                            if cid_i in state_by_id:
                                state_by_id[cid_i]["count"] += 1
                            if cid_i in selected_id_set:
                                total_made += 1

                        if total_made // cfg.PROGRESS_EVERY_N_OBJECTS != last_report_at // cfg.PROGRESS_EVERY_N_OBJECTS:
                            info("Прогресс (объекты): " + fmt_progress(total_made, total_target), end="\r")
                            last_report_at = total_made
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
                    if val_list is not None:
                        val_list.append(rel_img_path)
                    elif val_f is not None:
                        val_f.write(rel_img_path + "\n")
                else:
                    if train_list is not None:
                        train_list.append(rel_img_path)
                    elif train_f is not None:
                        train_f.write(rel_img_path + "\n")

                for (cid, xc, yc, ww, hh) in labels:
                    cid_i = int(cid)
                    ann_w.writerow([rel_img_path, cid, f"{xc:.6f}", f"{yc:.6f}", f"{ww:.6f}", f"{hh:.6f}"])
                    generated_counts[cid_i] = generated_counts.get(cid_i, 0) + 1
                    if cid_i in state_by_id:
                        state_by_id[cid_i]["count"] += 1
                    if cid_i in selected_id_set:
                        total_made += 1

                if total_made // cfg.PROGRESS_EVERY_N_OBJECTS != last_report_at // cfg.PROGRESS_EVERY_N_OBJECTS:
                    info("Прогресс (объекты): " + fmt_progress(total_made, total_target), end="\r")
                    last_report_at = total_made

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
                if val_list is not None:
                    val_list.append(rel_img_path)
                else:
                    # train/val уже открыт выше, но на всякий случай пишем безопасно
                    if val_f is not None:
                        val_f.write(rel_img_path + "\n")
                    else:
                        with open(os.path.join(out_dir, "val.txt"), "a", encoding="utf-8") as f:
                            f.write(rel_img_path + "\n")
            else:
                if train_list is not None:
                    train_list.append(rel_img_path)
                else:
                    if train_f is not None:
                        train_f.write(rel_img_path + "\n")
                    else:
                        with open(os.path.join(out_dir, "train.txt"), "a", encoding="utf-8") as f:
                            f.write(rel_img_path + "\n")

    write_dataset_yaml_and_splits(out_dir, all_classes, train_list, val_list)
    if train_f is not None:
        train_f.close()
    if val_f is not None:
        val_f.close()

    # Краткая сводка + подробный файл
    imported_total = int(sum(int(v) for v in imported_counts.values()))
    generated_total = int(sum(int(v) for v in generated_counts.values()))
    total_objects = imported_total + generated_total

    images_dir = os.path.join(out_dir, "images")
    labels_dir = os.path.join(out_dir, "labels")
    images_count = count_dir_files(images_dir, (".png", ".jpg", ".jpeg", ".bmp", ".webp"))
    label_files_count = count_dir_files(labels_dir, (".txt",))

    info("Статистика:")
    info(f"- Сколько объектов сгенерировано: {generated_total}")
    info(f"- Сколько объектов взято из внешнего датасета: {imported_total}")
    info(f"- Сколько всего объектов: {total_objects}")
    info(f"- Сколько изображений в датасете: {images_count}")
    info(f"- Сколько лейблов (label-файлов): {label_files_count}")

    write_data_stats(
        out_dir=out_dir,
        all_classes=all_classes,
        id_to_code=id_to_code,
        imported_counts=imported_counts,
        generated_counts=generated_counts,
        images_count=images_count,
        label_files_count=label_files_count,
    )

    info("Готово.")
    return out_dir


if __name__ == "__main__":
    generate_dataset()
