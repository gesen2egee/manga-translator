import os
import json
import base64
import argparse
import re
from io import BytesIO
from openai import OpenAI

TOKEN_PATTERN = re.compile(
    r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*|[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\u31f0-\u31ff\uac00-\ud7af]|[^\s]"
)
PUNCT_NO_SPACE_BEFORE = set(".,!?;:)]}。，、！？；：）」』】》〉’”")
PUNCT_NO_SPACE_AFTER = set("([{（「『【《〈‘“")

def encode_image_to_base64(image_path: str, max_size: int = 1024) -> str:
    """載入圖片並按比例將長邊縮放至 max_size，然後轉換為 Base64 字串供 API 使用"""
    try:
        from PIL import Image
    except ImportError:
        print("   [ERROR] 請先安裝 Pillow 套件才能處理圖片: pip install Pillow")
        return ""
        
    img = Image.open(image_path).convert('RGB')
    w, h = img.size
    
    if max(w, h) > max_size:
        # 計算縮放比例
        scale = max_size / float(max(w, h))
        new_w = int(w * scale)
        new_h = int(h * scale)
        # Pillow >= 9.1.0 支援 Image.Resampling.LANCZOS, 如果沒有則相容回 Image.LANCZOS
        resample_filter = getattr(Image, 'Resampling', Image).LANCZOS
        img = img.resize((new_w, new_h), resample_filter)
        
    # 存入記憶體並轉 Base64
    buffered = BytesIO()
    img.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def call_llm_for_manga_page(client: OpenAI, base64_image: str) -> str:
    """呼叫多模態模型取得翻譯與排版 JSON"""
    
    # 精簡版的 System Prompt (移除動態術語 RAG)
    system_prompt = """
你是一個頂級的漫畫在地化翻譯專家與排版設計師。
請找出圖片中所有文字，並回傳嚴格的 JSON 檔案。對於每一個對話框，你必須提供以下資訊：

1. `box_id`: 唯一的 ID，例如 "box_01"。
2. `original_text`: 從圖片上辨識出的原文（精確的 OCR）。
3. `translated_text`: 結合上下文與角色情緒，將原文翻譯為流暢生動的繁體中文。
4. `position`: 該對話框在圖片上的精確位置。
   ⚠️ 【極度重要座標規則】：
   你必須給出相對座標 `[x, y, w, h]`，其所有數值必須介於 0.0 到 1.0 之間。請提供 3 到 4 位小數點的精確度（例如 0.825 或 0.1234）。
   - `x`: 對話框中心的 X 座標比例 (從圖片左側算起)。
   - `y`: 對話框中心的 Y 座標比例 (從圖片上方算起)。
   - `w`: 對話框最大寬度的比例。
   - `h`: 對話框最大高度的比例。
5. `render_config`: 根據你的翻譯與對話框形狀，為台灣繁體中文設計的排版建議。
   - `alignment`: "left" (靠左/旁白), "center" (置中/常用), "right" (靠右)。
   - `direction`: "horizontal" (繁中預設橫排), "vertical" (僅限極度窄高的氣泡框使用)。
   - `font_size_offset`: 整數 (預設 0)。若畫面呈現大吼請給正整數 (+2 ~ +5)；若為小聲碎念給負數 (-1 ~ -3)。
   - `line_spacing`: 浮點數，行距，預設 0.2。
   - `fg_color`: [R, G, B]，多為黑字 [0, 0, 0]。若為深色底圖請改為白字 [255, 255, 255]。
   - `bg_color`: [R, G, B]，字體外層的描邊顏色。通常與字體顏色相反，確保文字清晰（例如黑字配白邊 [255, 255, 255]）。

確保整份輸出為有效的 JSON 格式，根節點必須包含 `dialogues` (陣列) 與 `global_suggestions` (物件)。
"""

    # 使用 OpenRouter 呼叫具備 推理 (Reasoning) 能力的模型
    response = client.chat.completions.create(
        model="qwen/qwen3.5-397b-a17b", # 根據使用者要求更改模型
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": "請分析這頁漫畫並輸出指定的 JSON 格式。"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]},
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": "請分析這頁漫畫並輸出指定的 JSON 格式。"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}
        ],
        response_format={"type": "json_object"}, # 強制輸出 JSON 格式
        extra_body={"reasoning": {"enabled": False}}, # 開啟 reasoning 以提高翻譯與座標精確度
        temperature=0.3 # 降低溫度以獲取穩定輸出
    )
    
    return response.choices[0].message.content

def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))

def _sanitize_rgb(color_value, default_color):
    if not isinstance(color_value, (list, tuple)) or len(color_value) < 3:
        return tuple(default_color)
    rgb = []
    for i in range(3):
        try:
            rgb.append(int(_clamp(float(color_value[i]), 0, 255)))
        except (TypeError, ValueError):
            rgb.append(default_color[i])
    return tuple(rgb)

def _measure_text(draw, text: str, font, stroke_width: int = 0):
    if not text:
        return 0, 0
    try:
        bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    except TypeError:
        bbox = draw.textbbox((0, 0), text, font=font)
    except AttributeError:
        size = draw.textsize(text, font=font)
        return size[0], size[1]
    return bbox[2] - bbox[0], bbox[3] - bbox[1]

def _line_height(font, draw):
    if hasattr(font, "getmetrics"):
        try:
            ascent, descent = font.getmetrics()
            line_h = ascent + descent
            if line_h > 0:
                return line_h
        except Exception:
            pass
    _, fallback_h = _measure_text(draw, "Ag測", font, stroke_width=0)
    return max(1, fallback_h)

def _font_with_size(font_path: str, base_font, font_size: int):
    try:
        from PIL import ImageFont
    except ImportError:
        return base_font

    if font_path and os.path.exists(font_path):
        try:
            return ImageFont.truetype(font_path, font_size)
        except Exception:
            pass
    if hasattr(base_font, "font_variant"):
        try:
            return base_font.font_variant(size=font_size)
        except Exception:
            pass
    return base_font

def _tokenize_line(text_line: str):
    return TOKEN_PATTERN.findall(text_line)

def _append_token(existing_text: str, token: str) -> str:
    if not existing_text:
        return token
    prev_char = existing_text[-1]
    if token and token[0] in PUNCT_NO_SPACE_BEFORE:
        separator = ""
    elif prev_char in PUNCT_NO_SPACE_AFTER:
        separator = ""
    elif prev_char.isalnum() and token and token[0].isalnum():
        separator = " "
    else:
        separator = ""
    return existing_text + separator + token

def normalize_position(position_data, image_width: int, image_height: int, min_box_px: int = 12):
    if isinstance(position_data, list) and len(position_data) >= 4:
        x, y, w, h = position_data[0], position_data[1], position_data[2], position_data[3]
    elif isinstance(position_data, dict):
        x = position_data.get("x", 0)
        y = position_data.get("y", 0)
        w = position_data.get("w", 0)
        h = position_data.get("h", 0)
    else:
        return None

    try:
        x = _clamp(float(x), 0.0, 1.0)
        y = _clamp(float(y), 0.0, 1.0)
        w = _clamp(float(w), 0.0, 1.0)
        h = _clamp(float(h), 0.0, 1.0)
    except (TypeError, ValueError):
        return None

    min_w_ratio = min_box_px / max(image_width, 1)
    min_h_ratio = min_box_px / max(image_height, 1)
    w = max(w, min_w_ratio)
    h = max(h, min_h_ratio)
    return x * image_width, y * image_height, w * image_width, h * image_height

def layout_horizontal(draw, text: str, font, max_width: int, max_height: int, line_spacing_px: int, stroke_width: int = 0):
    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = normalized.split("\n")
    lines = []

    for paragraph in paragraphs:
        if paragraph == "":
            lines.append("")
            continue
        tokens = _tokenize_line(paragraph)
        if not tokens:
            lines.append("")
            continue

        current_line = ""
        for token in tokens:
            candidate = _append_token(current_line, token)
            candidate_w, _ = _measure_text(draw, candidate, font, stroke_width=stroke_width)
            if current_line and candidate_w > max_width:
                lines.append(current_line)
                current_line = token
            else:
                current_line = candidate
        lines.append(current_line)

    if not lines:
        lines = [""]

    line_widths = [_measure_text(draw, line, font, stroke_width=stroke_width)[0] for line in lines]
    line_h = _line_height(font, draw)
    text_w = max(line_widths) if line_widths else 0
    text_h = line_h * len(lines) + max(0, len(lines) - 1) * line_spacing_px

    return {
        "direction": "horizontal",
        "lines": lines,
        "line_widths": line_widths,
        "line_height": line_h,
        "text_w": text_w,
        "text_h": text_h,
        "fits": text_w <= max_width and text_h <= max_height,
    }

def layout_vertical(draw, text: str, font, max_width: int, max_height: int, line_spacing_px: int, stroke_width: int = 0):
    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = normalized.split("\n")
    row_gap = max(1, int(font.size * 0.05))
    col_gap = max(1, line_spacing_px)
    _, sample_h = _measure_text(draw, "測", font, stroke_width=stroke_width)
    sample_h = max(1, sample_h)
    max_rows = max(1, int((max_height + row_gap) // (sample_h + row_gap)))

    columns = []
    for paragraph in paragraphs:
        chars = [ch for ch in paragraph if ch not in {" ", "\t"}]
        if not chars:
            continue
        current_col = []
        for ch in chars:
            current_col.append(ch)
            if len(current_col) >= max_rows:
                columns.append(current_col)
                current_col = []
        if current_col:
            columns.append(current_col)

    if not columns and normalized.strip():
        columns = [[ch for ch in normalized if ch not in {" ", "\n", "\t"}]]

    column_sizes = []
    column_widths = []
    column_heights = []
    for col in columns:
        sizes = []
        col_w = 0
        col_h = 0
        for idx, ch in enumerate(col):
            ch_w, ch_h = _measure_text(draw, ch, font, stroke_width=stroke_width)
            ch_w = max(1, ch_w)
            ch_h = max(1, ch_h)
            sizes.append((ch_w, ch_h))
            col_w = max(col_w, ch_w)
            col_h += ch_h
            if idx < len(col) - 1:
                col_h += row_gap
        column_sizes.append(sizes)
        column_widths.append(col_w)
        column_heights.append(col_h)

    text_w = sum(column_widths) + max(0, len(column_widths) - 1) * col_gap
    text_h = max(column_heights) if column_heights else 0

    return {
        "direction": "vertical",
        "columns": columns,
        "column_sizes": column_sizes,
        "column_widths": column_widths,
        "column_heights": column_heights,
        "row_gap": row_gap,
        "col_gap": col_gap,
        "text_w": text_w,
        "text_h": text_h,
        "fits": text_w <= max_width and text_h <= max_height,
    }

def _evaluate_layout_for_size(
    draw,
    text: str,
    direction: str,
    font_size: int,
    font_path: str,
    base_font,
    max_width: int,
    max_height: int,
    line_spacing_ratio: float,
    fit_padding: float,
):
    font = _font_with_size(font_path, base_font, font_size)
    stroke_width = max(1, int(font_size * 0.08))
    line_spacing_px = max(1, int(font_size * max(line_spacing_ratio, 0.01)))
    if direction == "vertical":
        layout = layout_vertical(
            draw,
            text,
            font,
            max_width=max_width,
            max_height=max_height,
            line_spacing_px=line_spacing_px,
            stroke_width=stroke_width,
        )
    else:
        layout = layout_horizontal(
            draw,
            text,
            font,
            max_width=max_width,
            max_height=max_height,
            line_spacing_px=line_spacing_px,
            stroke_width=stroke_width,
        )

    allowed_w = max(1.0, float(max_width) * fit_padding)
    allowed_h = max(1.0, float(max_height) * fit_padding)
    fit_score = max(layout["text_w"] / allowed_w, layout["text_h"] / allowed_h)

    layout["font"] = font
    layout["font_size"] = font_size
    layout["line_spacing_px"] = line_spacing_px
    layout["stroke_width"] = stroke_width
    layout["fit_score"] = fit_score
    layout["fits"] = fit_score <= 1.0
    return layout

def fit_font_size_binary_search(
    draw,
    text: str,
    direction: str,
    max_width: int,
    max_height: int,
    font_path: str,
    base_font,
    line_spacing_ratio: float,
    fit_padding: float,
    font_min: int,
    font_max: int,
):
    left = max(6, int(font_min))
    right = max(left, int(font_max))
    best_layout = None

    while left <= right:
        mid = (left + right) // 2
        layout = _evaluate_layout_for_size(
            draw,
            text,
            direction,
            mid,
            font_path,
            base_font,
            max_width,
            max_height,
            line_spacing_ratio,
            fit_padding,
        )
        if layout["fits"]:
            best_layout = layout
            left = mid + 1
        else:
            right = mid - 1

    if best_layout is None:
        best_layout = _evaluate_layout_for_size(
            draw,
            text,
            direction,
            max(6, int(font_min)),
            font_path,
            base_font,
            max_width,
            max_height,
            line_spacing_ratio,
            fit_padding,
        )
    return best_layout

def choose_direction_with_fallback(
    draw,
    text: str,
    preferred_direction: str,
    max_width: int,
    max_height: int,
    font_path: str,
    base_font,
    line_spacing_ratio: float,
    fit_padding: float,
    font_min: int,
    font_max: int,
):
    pref = (preferred_direction or "").strip().lower()
    if pref in {"v", "vertical"}:
        pref = "vertical"
    elif pref in {"h", "horizontal"}:
        pref = "horizontal"
    else:
        pref = "vertical" if max_height > max_width * 1.25 else "horizontal"

    alt = "vertical" if pref == "horizontal" else "horizontal"
    pref_layout = fit_font_size_binary_search(
        draw,
        text,
        pref,
        max_width,
        max_height,
        font_path,
        base_font,
        line_spacing_ratio,
        fit_padding,
        font_min,
        font_max,
    )
    alt_layout = fit_font_size_binary_search(
        draw,
        text,
        alt,
        max_width,
        max_height,
        font_path,
        base_font,
        line_spacing_ratio,
        fit_padding,
        font_min,
        font_max,
    )

    pref_fits = pref_layout["fit_score"] <= 1.0
    alt_fits = alt_layout["fit_score"] <= 1.0

    if pref_fits and not alt_fits:
        chosen = pref_layout
    elif alt_fits and not pref_fits:
        chosen = alt_layout
    elif pref_fits and alt_fits:
        if alt_layout["fit_score"] + 0.20 < pref_layout["fit_score"]:
            chosen = alt_layout
        else:
            chosen = pref_layout
    else:
        if pref_layout["fit_score"] <= alt_layout["fit_score"] + 0.20:
            chosen = pref_layout
        else:
            chosen = alt_layout

    chosen["preferred_direction"] = pref
    chosen["switched"] = chosen["direction"] != pref
    return chosen

def _compute_text_origin(box_rect, text_w: int, text_h: int, alignment: str, direction: str):
    left, top, right, bottom = box_rect
    box_w = max(1, right - left)
    box_h = max(1, bottom - top)

    if direction == "horizontal":
        if alignment == "left":
            x = left
        elif alignment == "right":
            x = right - text_w
        else:
            x = left + (box_w - text_w) / 2
    else:
        x = left + (box_w - text_w) / 2

    y = top + (box_h - text_h) / 2

    if text_w <= box_w:
        x = _clamp(x, left, right - text_w)
    if text_h <= box_h:
        y = _clamp(y, top, bottom - text_h)
    return int(round(x)), int(round(y))

def compute_text_erase_bbox(
    box_rect,
    text_rect,
    font_size: int,
    image_size,
    pad_x_scale: float = 0.35,
    pad_y_scale: float = 0.25,
):
    img_w, img_h = image_size
    box_left, box_top, box_right, box_bottom = box_rect
    text_left, text_top, text_right, text_bottom = text_rect
    pad_x = max(2, int(font_size * pad_x_scale))
    pad_y = max(2, int(font_size * pad_y_scale))

    erase_left = max(box_left, text_left - pad_x)
    erase_top = max(box_top, text_top - pad_y)
    erase_right = min(box_right, text_right + pad_x)
    erase_bottom = min(box_bottom, text_bottom + pad_y)

    erase_left = int(_clamp(erase_left, 0, img_w - 1))
    erase_top = int(_clamp(erase_top, 0, img_h - 1))
    erase_right = int(_clamp(erase_right, erase_left + 1, img_w))
    erase_bottom = int(_clamp(erase_bottom, erase_top + 1, img_h))
    return erase_left, erase_top, erase_right, erase_bottom

def _estimate_background_fill(source_img, rect):
    from PIL import ImageStat

    left, top, right, bottom = rect
    if right <= left or bottom <= top:
        return (255, 255, 255, 255)
    crop = source_img.crop((left, top, right, bottom)).convert("RGB")
    if crop.size[0] == 0 or crop.size[1] == 0:
        return (255, 255, 255, 255)
    mean = ImageStat.Stat(crop).mean
    return (int(mean[0]), int(mean[1]), int(mean[2]), 255)

def _draw_text_compat(draw, xy, text, font, fill, stroke_fill, stroke_width):
    try:
        draw.text(xy, text, font=font, fill=fill, stroke_width=stroke_width, stroke_fill=stroke_fill)
    except TypeError:
        draw.text(xy, text, font=font, fill=fill)

def _draw_horizontal_layout(draw, layout, origin_x: int, origin_y: int, alignment: str, fg_color, bg_color):
    line_y = origin_y
    text_w = layout["text_w"]
    for line, line_w in zip(layout["lines"], layout["line_widths"]):
        if alignment == "left":
            line_x = origin_x
        elif alignment == "right":
            line_x = origin_x + (text_w - line_w)
        else:
            line_x = origin_x + (text_w - line_w) / 2
        _draw_text_compat(
            draw,
            (line_x, line_y),
            line,
            layout["font"],
            fg_color,
            bg_color,
            layout["stroke_width"],
        )
        line_y += layout["line_height"] + layout["line_spacing_px"]

def _draw_vertical_layout(draw, layout, origin_x: int, origin_y: int, fg_color, bg_color):
    x_cursor = origin_x + layout["text_w"]
    for idx, col in enumerate(layout["columns"]):
        col_w = layout["column_widths"][idx]
        col_h = layout["column_heights"][idx]
        sizes = layout["column_sizes"][idx]
        x_cursor -= col_w
        y_cursor = origin_y + (layout["text_h"] - col_h) / 2
        for ch, (ch_w, ch_h) in zip(col, sizes):
            ch_x = x_cursor + (col_w - ch_w) / 2
            _draw_text_compat(
                draw,
                (ch_x, y_cursor),
                ch,
                layout["font"],
                fg_color,
                bg_color,
                layout["stroke_width"],
            )
            y_cursor += ch_h + layout["row_gap"]
        x_cursor -= layout["col_gap"]

def render_text_on_image(image_path: str, result_data: dict, output_path: str):
    """
    增強渲染器：
    1) 先依框尺寸與方向策略自動找可容納字級與排版
    2) 先擦「實際文字區」再上字，避免整框白底
    3) 支援直排/橫排、描邊與方向仲裁
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("   [ERROR] 請先安裝 Pillow 套件才能輸出圖片: pip install Pillow")
        return

    try:
        img = Image.open(image_path).convert("RGBA")
        source_img = img.copy()
        draw = ImageDraw.Draw(img)
        img_w, img_h = img.size

        font_path = "C:\\Windows\\Fonts\\msjh.ttc"
        base_font = None
        try:
            if os.path.exists(font_path):
                base_font = ImageFont.truetype(font_path, 20)
        except Exception as e:
            print(f"   [WARN] 字體加載失敗 ({e})")
             
        if base_font is None:
            base_font = ImageFont.load_default()
            print("   [WARN] 找不到微軟正黑體，將使用系統預設字體（中文可能變方塊）。")

        dialogues = result_data.get("dialogues", [])
        render_jobs = []

        for dlg in dialogues:
            normalized = normalize_position(dlg.get("position"), img_w, img_h)
            if normalized is None:
                continue
            center_x, center_y, box_w, box_h = normalized
            left = int(_clamp(round(center_x - box_w / 2), 0, img_w - 1))
            top = int(_clamp(round(center_y - box_h / 2), 0, img_h - 1))
            right = int(_clamp(round(center_x + box_w / 2), left + 1, img_w))
            bottom = int(_clamp(round(center_y + box_h / 2), top + 1, img_h))
            if right - left < 2 or bottom - top < 2:
                continue

            config = dlg.get("render_config") or {}
            text = str(dlg.get("translated_text", "")).strip()
            if not text:
                continue

            alignment = str(config.get("alignment", "center")).lower()
            if alignment not in {"left", "center", "right"}:
                alignment = "center"

            try:
                line_spacing_ratio = float(config.get("line_spacing", 0.2))
            except (TypeError, ValueError):
                line_spacing_ratio = 0.2
            line_spacing_ratio = _clamp(line_spacing_ratio, 0.01, 1.0)

            fg_color = _sanitize_rgb(config.get("fg_color"), (0, 0, 0))
            bg_color = _sanitize_rgb(config.get("bg_color"), (255, 255, 255))

            font_min = 8
            font_max = max(font_min + 2, min(128, int(max(box_w, box_h) * 0.9)))
            fit_padding = 1.00

            chosen = choose_direction_with_fallback(
                draw,
                text,
                str(config.get("direction", "")),
                right - left,
                bottom - top,
                font_path,
                base_font,
                line_spacing_ratio,
                fit_padding,
                font_min,
                font_max,
            )

            try:
                offset = int(config.get("font_size_offset") or 0)
            except (TypeError, ValueError):
                offset = 0
            if offset != 0:
                target_size = int(_clamp(chosen["font_size"] + offset * 2, font_min, font_max))
                if target_size != chosen["font_size"]:
                    adjusted = _evaluate_layout_for_size(
                        draw,
                        text,
                        chosen["direction"],
                        target_size,
                        font_path,
                        base_font,
                        right - left,
                        bottom - top,
                        line_spacing_ratio,
                        fit_padding,
                    )
                    if offset < 0:
                        chosen = adjusted
                    elif offset > 0 and adjusted["fit_score"] <= 1.02:
                        chosen = adjusted

            origin_x, origin_y = _compute_text_origin(
                (left, top, right, bottom),
                chosen["text_w"],
                chosen["text_h"],
                alignment,
                chosen["direction"],
            )
            text_rect = (
                origin_x,
                origin_y,
                origin_x + chosen["text_w"],
                origin_y + chosen["text_h"],
            )
            erase_rect = compute_text_erase_bbox(
                (left, top, right, bottom),
                text_rect,
                chosen["font_size"],
                (img_w, img_h),
            )

            box_id = dlg.get("box_id", "unknown")
            if chosen.get("switched"):
                print(
                    f"   [INFO] [方向回退] {box_id}: LLM={chosen['preferred_direction']} -> 使用 {chosen['direction']}"
                )
            if chosen["fit_score"] > 1.0:
                print(
                    f"   [WARN] [排版警告] {box_id}: direction={chosen['direction']} font={chosen['font_size']} fit={chosen['fit_score']:.2f}"
                )

            render_jobs.append(
                {
                    "layout": chosen,
                    "alignment": alignment,
                    "fg_color": fg_color,
                    "bg_color": bg_color,
                    "erase_rect": erase_rect,
                    "erase_fill": _estimate_background_fill(source_img, erase_rect),
                    "origin": (origin_x, origin_y),
                }
            )

        # pass 1: 先擦除文字區（只擦實際文字 bbox）
        for job in render_jobs:
            left, top, right, bottom = job["erase_rect"]
            radius = max(1, int(job["layout"]["font_size"] * 0.12))
            if hasattr(draw, "rounded_rectangle"):
                draw.rounded_rectangle([left, top, right, bottom], radius=radius, fill=job["erase_fill"])
            else:
                draw.rectangle([left, top, right, bottom], fill=job["erase_fill"])

        # pass 2: 再上字，避免後續擦底覆蓋先前文字
        for job in render_jobs:
            layout = job["layout"]
            origin_x, origin_y = job["origin"]
            if layout["direction"] == "vertical":
                _draw_vertical_layout(
                    draw,
                    layout,
                    origin_x,
                    origin_y,
                    job["fg_color"],
                    job["bg_color"],
                )
            else:
                _draw_horizontal_layout(
                    draw,
                    layout,
                    origin_x,
                    origin_y,
                    job["alignment"],
                    job["fg_color"],
                    job["bg_color"],
                )

        if output_path.lower().endswith(".jpg") or output_path.lower().endswith(".jpeg"):
            img = img.convert("RGB")

        img.save(output_path)
        print(f"   [OK] 圖片已渲染並匯出至 {output_path}")

    except Exception as e:
        print(f"   [ERROR] 渲染圖片發生錯誤：{e}")


def process_directory(input_dir: str, output_dir: str):
    """處理整個資料夾的批次邏輯"""
    
    # 確保輸出資料夾存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 從 OS 環保變數讀取 API Key
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("[ERROR] 找不到 OPENROUTER_API_KEY 系統變數。")
        print("請先在終端機執行：set OPENROUTER_API_KEY=你的金鑰")
        return
         
    print("[OK] 成功讀取 API Key，初始化連線中...")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    
    # 搜尋所有 jpg/png 檔案
    extensions = (".jpg", ".jpeg", ".png")
    image_paths = []
    
    # 使用 os.listdir 避免 glob.glob 無法辨識中括號 [] 等特殊字元的問題
    if os.path.exists(input_dir):
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(extensions):
                image_paths.append(os.path.join(input_dir, filename))
    
    # 確保按照字母順序排序
    image_paths.sort()
    
    if not image_paths:
        print(f"[WARN] 在 {input_dir} 中找不到任何圖片檔案。")
        return

    print(f"[INFO] 找到 {len(image_paths)} 張圖片，開始批次處理...\n")
    
    for img_path in image_paths:
        base_name = os.path.basename(img_path)
        print(f"[INFO] [{base_name}] 開始處理...")
        
        try:
            # 1. 讀圖編碼
            encoded_img = encode_image_to_base64(img_path)
            if not encoded_img:
                print(f"   [ERROR] 無法讀取或編碼圖片，已跳過 {base_name}\n")
                continue
            
            # 2. 呼叫 LLM
            print(f"   [INFO] 正在進行深度推理與翻譯，這可能需要幾十秒...")
            json_result_str = call_llm_for_manga_page(client, encoded_img)
            
            # 3. 解析 JSON
            result_data = json.loads(json_result_str)
            
            # 4. 儲存 JSON 至目的資料夾
            file_name_without_ext = os.path.splitext(base_name)[0]
            output_json_path = os.path.join(output_dir, f"{file_name_without_ext}.json")
            
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
                
            # 5. 呼叫簡易渲染器，產生翻譯後的圖片
            output_img_path = os.path.join(output_dir, f"{file_name_without_ext}_translated.png")
            render_text_on_image(img_path, result_data, output_img_path)
                
            print(f"   [OK] JSON 存至 {output_json_path}\n")
            
        except json.JSONDecodeError:
            print(f"   [ERROR] JSON 解析失敗，模型回傳的格式不正確！")
            # 也可以把錯誤字串存下來方便 debug
            debug_path = os.path.join(output_dir, f"{os.path.basename(img_path)}_error.txt")
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(json_result_str)
        except Exception as e:
            print(f"   [ERROR] 處理發生未知錯誤：{str(e)}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多模態 LLM 漫畫翻譯 CLI 工具")
    parser.add_argument("--input", "-i", type=str, required=True, help="輸入漫畫圖片的資料夾路徑")
    parser.add_argument("--output", "-o", type=str, required=True, help="最終輸出 JSON 的資料夾路徑")
    
    args = parser.parse_args()
    process_directory(args.input, args.output)
