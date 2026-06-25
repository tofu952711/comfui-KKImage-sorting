import json

import cv2
import numpy as np
import torch


def _as_uint8_rgb(image):
    image = image.detach().float().clamp(0.0, 1.0).cpu().numpy()
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.shape[-1] > 3:
        image = image[..., :3]
    return (image * 255.0 + 0.5).astype(np.uint8)


def _odd_kernel(value):
    value = max(1, int(round(value)))
    return value if value % 2 == 1 else value + 1


def _background_color(rgb, border_percent):
    height, width = rgb.shape[:2]
    border = max(2, int(round(min(height, width) * float(border_percent) / 100.0)))
    border = max(1, min(border, height, width))

    samples = np.concatenate(
        [
            rgb[:border, :, :].reshape(-1, 3),
            rgb[-border:, :, :].reshape(-1, 3),
            rgb[:, :border, :].reshape(-1, 3),
            rgb[:, -border:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    return np.median(samples.astype(np.float32), axis=0)


def _edge_touch_fraction(mask):
    height, width = mask.shape
    edge_pixels = (
        np.count_nonzero(mask[0, :])
        + np.count_nonzero(mask[-1, :])
        + np.count_nonzero(mask[:, 0])
        + np.count_nonzero(mask[:, -1])
    )
    return edge_pixels / max(1, (height + width) * 2)


def _center_weight(left, top, width, height, image_width, image_height):
    cx = (left + width * 0.5) / max(1, image_width)
    cy = (top + height * 0.5) / max(1, image_height)
    distance = ((cx - 0.5) ** 2 + (cy - 0.5) ** 2) ** 0.5
    return max(0.0, 1.0 - distance / 0.70710678)


def _score_image(
    image,
    threshold,
    min_area_percent,
    close_radius,
    border_percent,
    edge_penalty,
    center_bias,
):
    rgb = _as_uint8_rgb(image)
    height, width = rgb.shape[:2]
    total_area = max(1, height * width)

    bg = _background_color(rgb, border_percent)
    rgb_float = rgb.astype(np.float32)
    bg_distance = np.linalg.norm(rgb_float - bg.reshape(1, 1, 3), axis=2)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1].astype(np.float32)
    value = hsv[:, :, 2].astype(np.float32)

    threshold = float(threshold)
    value_floor = max(3.0, threshold * 0.45)
    color_score = bg_distance + saturation * 0.22 + np.maximum(value - value_floor, 0.0) * 0.12
    mask = (color_score >= threshold).astype(np.uint8) * 255

    close_size = _odd_kernel(close_radius)
    if close_size > 1:
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    open_size = max(1, _odd_kernel(close_size // 3))
    if open_size > 1:
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    min_area = max(1, int(round(total_area * float(min_area_percent) / 100.0)))

    kept_mask = np.zeros_like(mask)
    components = []
    for label in range(1, component_count):
        left, top, comp_width, comp_height, area = stats[label]
        if area < min_area:
            continue

        bbox_area = max(1, int(comp_width) * int(comp_height))
        fill_ratio = float(area) / float(bbox_area)
        aspect = max(float(comp_width) / max(1.0, float(comp_height)), float(comp_height) / max(1.0, float(comp_width)))

        # Thin, sparse marks often come from sparks, frame lines, or UI edges.
        if aspect > 18.0 and fill_ratio < 0.18:
            continue

        component_mask = labels == label
        kept_mask[component_mask] = 255
        components.append(
            {
                "area": int(area),
                "bbox_area": int(bbox_area),
                "fill_ratio": float(fill_ratio),
                "aspect": float(aspect),
                "center": _center_weight(left, top, comp_width, comp_height, width, height),
            }
        )

    subject_area = int(np.count_nonzero(kept_mask))
    if subject_area == 0 or not components:
        return {
            "score": 0.0,
            "subject_area_percent": 0.0,
            "largest_component_percent": 0.0,
            "bbox_area_percent": 0.0,
            "component_count": 0,
            "edge_touch": 0.0,
        }

    largest = max(components, key=lambda item: item["area"])
    subject_ratio = subject_area / total_area
    largest_ratio = largest["area"] / total_area
    bbox_ratio = largest["bbox_area"] / total_area
    component_bonus = min(len(components), 6) * 0.012
    center_bonus = largest["center"] * float(center_bias) * 0.08
    edge_touch = _edge_touch_fraction(kept_mask)

    sparse_bbox_penalty = max(0.0, bbox_ratio - subject_ratio * 3.8) * 0.12
    score = (
        subject_ratio * 1.45
        + largest_ratio * 0.65
        + min(bbox_ratio, subject_ratio * 2.5) * 0.18
        + component_bonus
        + center_bonus
        - edge_touch * float(edge_penalty) * 0.35
        - sparse_bbox_penalty
    )

    return {
        "score": float(max(0.0, score)),
        "subject_area_percent": float(subject_ratio * 100.0),
        "largest_component_percent": float(largest_ratio * 100.0),
        "bbox_area_percent": float(bbox_ratio * 100.0),
        "component_count": int(len(components)),
        "edge_touch": float(edge_touch),
    }


class SubjectAreaSelector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像列表": ("IMAGE",),
                "前景阈值": (
                    "FLOAT",
                    {"default": 34.0, "min": 1.0, "max": 160.0, "step": 1.0},
                ),
                "最小碎片面积%": (
                    "FLOAT",
                    {"default": 0.05, "min": 0.0, "max": 5.0, "step": 0.01},
                ),
                "连接半径": (
                    "INT",
                    {"default": 9, "min": 1, "max": 101, "step": 2},
                ),
                "背景采样边缘%": (
                    "FLOAT",
                    {"default": 4.0, "min": 1.0, "max": 25.0, "step": 0.5},
                ),
                "边缘残片惩罚": (
                    "FLOAT",
                    {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
                "中心偏好": (
                    "FLOAT",
                    {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("最佳图像", "最佳索引", "最佳评分", "全部评分")
    FUNCTION = "select"
    CATEGORY = "图像/筛选"

    @staticmethod
    def _rank(inputs):
        images = inputs["图像列表"]
        if images.ndim == 3:
            images = images.unsqueeze(0)

        scores = []
        for index in range(images.shape[0]):
            detail = _score_image(
                images[index],
                threshold=inputs["前景阈值"],
                min_area_percent=inputs["最小碎片面积%"],
                close_radius=inputs["连接半径"],
                border_percent=inputs["背景采样边缘%"],
                edge_penalty=inputs["边缘残片惩罚"],
                center_bias=inputs["中心偏好"],
            )
            detail["index"] = int(index)
            scores.append(detail)

        ranked = sorted(scores, key=lambda item: item["score"], reverse=True)
        return images, scores, ranked

    def select(self, **inputs):
        images, scores, ranked = self._rank(inputs)
        best = ranked[0]
        best_index = int(best["index"])
        best_image = images[best_index : best_index + 1].clone()
        scores_text = json.dumps(scores, ensure_ascii=False, separators=(",", ":"))

        return (best_image, best_index, float(best["score"]), scores_text)


class SubjectAreaSorter(SubjectAreaSelector):
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("排序后图像列表", "排序索引", "全部评分")
    FUNCTION = "sort"
    CATEGORY = "图像/筛选"

    def sort(self, **inputs):
        images, scores, ranked = self._rank(inputs)
        indices = [int(item["index"]) for item in ranked]
        sorted_images = images[indices].clone()
        index_text = json.dumps(indices, ensure_ascii=False, separators=(",", ":"))
        scores_text = json.dumps(scores, ensure_ascii=False, separators=(",", ":"))
        return (sorted_images, index_text, scores_text)


NODE_CLASS_MAPPINGS = {
    "KKSubjectAreaSelector": SubjectAreaSelector,
    "KKSubjectAreaSorter": SubjectAreaSorter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KKSubjectAreaSelector": "KK 主体占比筛选",
    "KKSubjectAreaSorter": "KK主体占比排序",
}
