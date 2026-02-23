"""LLM client utilities: image encoding, upstream error detection, and OpenRouter JSON generation."""

import base64
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
