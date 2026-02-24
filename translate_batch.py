"""Batch entrypoint for manga translation: orchestrates LLM call, JSON save, and image rendering."""

import argparse
import json
import math
import os

from openai import OpenAI

from llm_client import call_llm_for_manga_batch, encode_image_to_base64, looks_like_html_error
from renderer import render_text_on_image


def process_directory(input_dir: str, output_dir: str, max_batch_size: int = 8):
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

    if max_batch_size < 1:
        print(f"[WARN] max_batch_size={max_batch_size} 不合法，已改用 1")
        max_batch_size = 1

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

    total_batches = math.ceil(len(image_paths) / max_batch_size)
    print(
        f"[INFO] 找到 {len(image_paths)} 張圖片，"
        f"每批最多 {max_batch_size} 張，總批次數 {total_batches}，開始處理...\n"
    )

    for batch_idx, start in enumerate(range(0, len(image_paths), max_batch_size), start=1):
        batch_paths = image_paths[start : start + max_batch_size]
        print(
            f"[INFO] [BATCH {batch_idx}/{total_batches}] "
            f"開始，圖片數量: {len(batch_paths)}"
        )

        encoded_images = []
        valid_paths = []
        for img_path in batch_paths:
            base_name = os.path.basename(img_path)
            print(f"   [INFO] [{base_name}] 讀圖與編碼中...")
            encoded_img = encode_image_to_base64(img_path)
            if not encoded_img:
                print(f"   [ERROR] 無法讀取或編碼圖片，已跳過 {base_name}")
                continue
            encoded_images.append(encoded_img)
            valid_paths.append(img_path)

        if not encoded_images:
            print(f"[WARN] [BATCH {batch_idx}/{total_batches}] 沒有可用圖片，略過此批次。\n")
            continue

        try:
            print(
                f"[INFO] [BATCH {batch_idx}/{total_batches}] "
                f"送出 {len(encoded_images)} 張圖片至 LLM 批次翻譯..."
            )
            batch_results = call_llm_for_manga_batch(
                client=client,
                base64_images=encoded_images,
                image_labels=[os.path.basename(path) for path in valid_paths],
            )
        except Exception as e:
            error_text = str(e)
            print(f"[ERROR] [BATCH {batch_idx}/{total_batches}] 批次翻譯失敗：{error_text}\n")
            if looks_like_html_error(error_text):
                debug_path = os.path.join(output_dir, f"batch_{batch_idx:04d}_http_error.html")
                with open(debug_path, "w", encoding="utf-8") as f:
                    f.write(error_text)
                print(f"[DEBUG] 已保存 HTML 錯誤頁至 {debug_path}\n")
            continue

        if len(batch_results) != len(valid_paths):
            print(
                f"[WARN] [BATCH {batch_idx}/{total_batches}] "
                f"回傳筆數({len(batch_results)})與圖片數({len(valid_paths)})不一致，"
                "將以最短長度繼續處理。"
            )

        for img_path, result_data in zip(valid_paths, batch_results):
            base_name = os.path.basename(img_path)
            print(f"   [INFO] [{base_name}] 開始寫入結果...")

            try:
                if not isinstance(result_data, dict):
                    raise ValueError("result_data 不是 dict")

                dialogues = result_data.get("dialogues", [])
                if not isinstance(dialogues, list):
                    dialogues = []
                result_data = {"dialogues": dialogues}

                dialogues_count = len(dialogues)
                print(f"   [INFO] [JSON] 解析成功，dialogues 數量: {dialogues_count}")
                if dialogues_count > 0 and isinstance(dialogues[0], dict):
                    print(f"   [DEBUG] [JSON] 第一筆欄位: {list(dialogues[0].keys())}")

                file_name_without_ext = os.path.splitext(base_name)[0]
                output_json_path = os.path.join(output_dir, f"{file_name_without_ext}.json")

                with open(output_json_path, "w", encoding="utf-8") as f:
                    json.dump(result_data, f, ensure_ascii=False, indent=2)

                output_img_path = os.path.join(output_dir, f"{file_name_without_ext}_translated.png")
                render_text_on_image(img_path, result_data, output_img_path)

                print(f"   [OK] JSON 存至 {output_json_path}\n")
            except Exception as e:
                error_text = str(e)
                print(f"   [ERROR] [{base_name}] 處理發生未知錯誤：{error_text}\n")
                if looks_like_html_error(error_text):
                    debug_path = os.path.join(output_dir, f"{base_name}_http_error.html")
                    with open(debug_path, "w", encoding="utf-8") as f:
                        f.write(error_text)
                    print(f"   [DEBUG] 已保存 HTML 錯誤頁至 {debug_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多模態 LLM 漫畫翻譯 CLI 工具")
    parser.add_argument("--input", "-i", type=str, required=True, help="輸入漫畫圖片的資料夾路徑")
    parser.add_argument("--output", "-o", type=str, required=True, help="最終輸出 JSON 的資料夾路徑")
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=8,
        help="每次送入 LLM 的最大圖片張數（預設: 8）",
    )

    args = parser.parse_args()
    process_directory(args.input, args.output, max_batch_size=args.max_batch_size)
