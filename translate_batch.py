"""Batch entrypoint for manga translation: orchestrates LLM call, JSON save, and image rendering."""

import argparse
import json
import os

from openai import OpenAI

from llm_client import call_llm_for_manga_page, encode_image_to_base64, looks_like_html_error
from renderer import render_text_on_image


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
        json_result_str = ""
        
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
            dialogues_count = len(result_data.get("dialogues", []))
            print(f"   [INFO] [JSON] 解析成功，dialogues 數量: {dialogues_count}")
            if dialogues_count > 0:
                first_dialogue = result_data["dialogues"][0]
                print(f"   [DEBUG] [JSON] 第一筆欄位: {list(first_dialogue.keys())}")
            
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
            if json_result_str:
                print(f"   [DEBUG] [LLM] 解析失敗時的原始輸出字元數: {len(json_result_str)}")
                print("   [DEBUG] [LLM] 解析失敗時原始輸出 BEGIN")
                print(json_result_str)
                print("   [DEBUG] [LLM] 解析失敗時原始輸出 END")
            # 也可以把錯誤字串存下來方便 debug
            debug_path = os.path.join(output_dir, f"{os.path.basename(img_path)}_error.txt")
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(json_result_str)
        except Exception as e:
            error_text = str(e)
            print(f"   [ERROR] 處理發生未知錯誤：{error_text}\n")
            if looks_like_html_error(error_text):
                debug_path = os.path.join(output_dir, f"{os.path.basename(img_path)}_http_error.html")
                with open(debug_path, "w", encoding="utf-8") as f:
                    f.write(error_text)
                print(f"   [DEBUG] 已保存 HTML 錯誤頁至 {debug_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多模態 LLM 漫畫翻譯 CLI 工具")
    parser.add_argument("--input", "-i", type=str, required=True, help="輸入漫畫圖片的資料夾路徑")
    parser.add_argument("--output", "-o", type=str, required=True, help="最終輸出 JSON 的資料夾路徑")
    
    args = parser.parse_args()
    process_directory(args.input, args.output)
