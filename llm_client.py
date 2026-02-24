"""LLM client utilities: image encoding, upstream error detection, and OpenRouter JSON generation."""

import base64
import json
import time
from io import BytesIO

from openai import OpenAI


def looks_like_html_error(text: str) -> bool:
    if not isinstance(text, str):
        return False
    lower_text = text.strip().lower()
    return "<!doctype html" in lower_text or "<html" in lower_text


def encode_image_to_base64(image_path: str, max_size: int = 1024) -> str:
    """載入圖片並按比例將長邊縮放至 max_size，然後轉換為 Base64 字串供 API 使用"""
    try:
        from PIL import Image
    except ImportError:
        print("   [ERROR] 請先安裝 Pillow 套件才能處理圖片: pip install Pillow")
        return ""

    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    if max(w, h) > max_size:
        scale = max_size / float(max(w, h))
        new_w = int(w * scale)
        new_h = int(h * scale)
        resample_filter = getattr(Image, "Resampling", Image).LANCZOS
        img = img.resize((new_w, new_h), resample_filter)

    buffered = BytesIO()
    img.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def call_llm_for_manga_page(
    client: OpenAI,
    base64_image: str,
    max_retries: int = 3,
    retry_base_seconds: float = 2.0,
) -> str:
    """呼叫多模態模型取得最小可渲染 JSON"""

    system_prompt = """
你是一個頂級的漫畫在地化翻譯專家。
請找出圖片中所有可翻譯文字，並回傳嚴格 JSON。

只允許以下結構與欄位：
{
  "dialogues": [
    {
      "translated_text": "流暢生動的台灣繁體中文",
      "position": {
        "corners": [
          [x1, y1],
          [x2, y2],
          [x3, y3],
          [x4, y4]
        ]
      }
    }
  ]
}

規則：
1. 四個角點都必須在 0.0 到 1.0 之間，保留 3 到 4 位小數。
2. `corners` 依照「左上、右上、右下、左下」順序輸出。
3. 僅輸出 `translated_text` 與 `position`，不要附加其他鍵。
"""

    print(f"   [INFO] [LLM] 準備請求模型，影像 base64 長度: {len(base64_image)}")

    last_error = ""
    for attempt in range(1, max_retries + 1):
        print(f"   [INFO] [LLM] 呼叫嘗試 {attempt}/{max_retries}")
        try:
            response = client.chat.completions.create(
                model="qwen/qwen3.5-397b-a17b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "請輸出最小可渲染 JSON。"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        ],
                    },
                ],
                response_format={"type": "json_object"},
                extra_body={"reasoning": {"enabled": False}},
                temperature=0.3,
            )
        except Exception as e:
            last_error = str(e)
            short_err = last_error if len(last_error) < 600 else last_error[:600] + " ...[truncated]"
            print(f"   [WARN] [LLM] API 呼叫失敗: {short_err}")
            if attempt < max_retries:
                wait_s = retry_base_seconds * attempt
                print(f"   [INFO] [LLM] {wait_s:.1f}s 後重試")
                time.sleep(wait_s)
                continue
            raise RuntimeError(f"LLM API 呼叫失敗（重試 {max_retries} 次）: {last_error}")

        usage = getattr(response, "usage", None)
        if usage is not None:
            prompt_tokens = getattr(usage, "prompt_tokens", "n/a")
            completion_tokens = getattr(usage, "completion_tokens", "n/a")
            total_tokens = getattr(usage, "total_tokens", "n/a")
            print(
                f"   [INFO] [LLM] token usage: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}"
            )

        llm_content = response.choices[0].message.content or ""
        print(f"   [INFO] [LLM] 回傳字元數: {len(llm_content)}")
        print("   [DEBUG] [LLM] 原始輸出 BEGIN")
        print(llm_content)
        print("   [DEBUG] [LLM] 原始輸出 END")

        if not llm_content.strip():
            last_error = "LLM 回傳空字串"
            print(f"   [WARN] [LLM] {last_error}")
        elif looks_like_html_error(llm_content):
            last_error = "LLM 回傳 HTML 錯誤頁（可能是上游 5xx）"
            print(f"   [WARN] [LLM] {last_error}")
        else:
            return llm_content

        if attempt < max_retries:
            wait_s = retry_base_seconds * attempt
            print(f"   [INFO] [LLM] {wait_s:.1f}s 後重試")
            time.sleep(wait_s)

    raise RuntimeError(f"LLM 回傳非預期內容（重試 {max_retries} 次）: {last_error}")


def call_llm_for_manga_batch(
    client: OpenAI,
    base64_images: list[str],
    image_labels: list[str] | None = None,
    max_retries: int = 3,
    retry_base_seconds: float = 2.0,
) -> list[dict]:
    """一次送入多張圖片，回傳每張圖片對應的最小可渲染 JSON。"""

    if not base64_images:
        return []

    system_prompt = """
你是一個頂級的漫畫在地化翻譯專家。
請找出圖片中所有可翻譯文字，並回傳嚴格 JSON。

只允許以下結構與欄位：
{
  "dialogues": [
    {
      "translated_text": "流暢生動的台灣繁體中文",
      "position": {
        "corners": [
          [x1, y1],
          [x2, y2],
          [x3, y3],
          [x4, y4]
        ]
      }
    }
  ]
}

規則：
1. 四個角點都必須在 0.0 到 1.0 之間，保留 3 到 4 位小數。
2. `corners` 依照「左上、右上、右下、左下」順序輸出。
3. 僅輸出 `translated_text` 與 `position`，不要附加其他鍵。
"""

    labels = image_labels if image_labels and len(image_labels) == len(base64_images) else None

    prefix_lines = [f"第 {idx + 1} 張：{name}" for idx, name in enumerate(labels)] if labels else []
    image_list_text = "\n".join(prefix_lines)

    front_system_text = f"【系統詞-前】\n{system_prompt}"
    back_system_text = f"【系統詞-後】\n{system_prompt}"

    user_content = [
        {"type": "text", "text": front_system_text},
        {
            "type": "text",
            "text": (
                f"以下共有 {len(base64_images)} 張圖片，請依輸入順序輸出 JSON。\n"
                f"{image_list_text}\n\n"
                "你只能回傳以下格式：\n"
                "{\n"
                '  "results": [\n'
                "    {\n"
                '      "index": 1,\n'
                '      "dialogues": [\n'
                "        {\n"
                '          "translated_text": "流暢生動的台灣繁體中文",\n'
                '          "position": {\n'
                '            "corners": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]\n'
                "          }\n"
                "        }\n"
                "      ]\n"
                "    }\n"
                "  ]\n"
                "}\n\n"
                "規則：\n"
                "1. results 必須與輸入圖片數量一致。\n"
                "2. index 從 1 開始，且必須對應輸入順序。\n"
                "3. 每筆 dialogues 僅可包含 translated_text 與 position。\n"
                "4. 不要輸出任何 JSON 以外的內容。"
            ),
        },
    ]

    for base64_image in base64_images:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            }
        )

    user_content.append({"type": "text", "text": back_system_text})

    print(f"   [INFO] [LLM-BATCH] 準備請求模型，影像數量: {len(base64_images)}")

    last_error = ""
    for attempt in range(1, max_retries + 1):
        print(f"   [INFO] [LLM-BATCH] 呼叫嘗試 {attempt}/{max_retries}")
        try:
            response = client.chat.completions.create(
                model="qwen/qwen3.5-397b-a17b",
                messages=[
                    {"role": "system", "content": "你必須嚴格輸出符合規範的 JSON。"},
                    {"role": "user", "content": user_content},
                ],
                response_format={"type": "json_object"},
                extra_body={"reasoning": {"enabled": False}},
                temperature=0.3,
            )
        except Exception as e:
            last_error = str(e)
            short_err = last_error if len(last_error) < 600 else last_error[:600] + " ...[truncated]"
            print(f"   [WARN] [LLM-BATCH] API 呼叫失敗: {short_err}")
            if attempt < max_retries:
                wait_s = retry_base_seconds * attempt
                print(f"   [INFO] [LLM-BATCH] {wait_s:.1f}s 後重試")
                time.sleep(wait_s)
                continue
            raise RuntimeError(f"LLM API 呼叫失敗（重試 {max_retries} 次）: {last_error}")

        usage = getattr(response, "usage", None)
        if usage is not None:
            prompt_tokens = getattr(usage, "prompt_tokens", "n/a")
            completion_tokens = getattr(usage, "completion_tokens", "n/a")
            total_tokens = getattr(usage, "total_tokens", "n/a")
            print(
                f"   [INFO] [LLM-BATCH] token usage: prompt={prompt_tokens}, "
                f"completion={completion_tokens}, total={total_tokens}"
            )

        llm_content = response.choices[0].message.content or ""
        print(f"   [INFO] [LLM-BATCH] 回傳字元數: {len(llm_content)}")
        print("   [DEBUG] [LLM-BATCH] 原始輸出 BEGIN")
        print(llm_content)
        print("   [DEBUG] [LLM-BATCH] 原始輸出 END")

        if not llm_content.strip():
            last_error = "LLM 回傳空字串"
            print(f"   [WARN] [LLM-BATCH] {last_error}")
        elif looks_like_html_error(llm_content):
            last_error = "LLM 回傳 HTML 錯誤頁（可能是上游 5xx）"
            print(f"   [WARN] [LLM-BATCH] {last_error}")
        else:
            try:
                payload = json.loads(llm_content)
                raw_results = payload.get("results", [])
                if not isinstance(raw_results, list):
                    raise ValueError("results 欄位不是陣列")

                normalized_results: list[dict] = [{"dialogues": []} for _ in base64_images]
                for item in raw_results:
                    if not isinstance(item, dict):
                        continue

                    idx_raw = item.get("index")
                    if isinstance(idx_raw, bool):
                        continue

                    try:
                        idx = int(idx_raw)
                    except (TypeError, ValueError):
                        continue

                    if not (1 <= idx <= len(base64_images)):
                        continue

                    dialogues = item.get("dialogues", [])
                    if not isinstance(dialogues, list):
                        dialogues = []

                    normalized_results[idx - 1] = {"dialogues": dialogues}

                return normalized_results
            except Exception as parse_error:
                last_error = f"LLM 回傳 JSON 結構不符合預期: {parse_error}"
                print(f"   [WARN] [LLM-BATCH] {last_error}")

        if attempt < max_retries:
            wait_s = retry_base_seconds * attempt
            print(f"   [INFO] [LLM-BATCH] {wait_s:.1f}s 後重試")
            time.sleep(wait_s)

    raise RuntimeError(f"LLM 回傳非預期內容（重試 {max_retries} 次）: {last_error}")
