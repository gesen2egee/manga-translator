---
description: [基於 OpenRouter API 的 CLI 批次漫畫翻譯流程設計]
---

# 基於 OpenRouter API 的 CLI 批次漫畫翻譯流程 (CLI Workflow)

使用 Python 撰寫命令列工具 (CLI)，透過 OpenRouter API 對本機圖片進行批次端到端漫畫翻譯與 JSON 輸出。

## 1. 系統架構與需求

我們將使用 `openai` Python 套件來連線至相容其格式的 OpenRouter 端點，並利用如 `qwen/qwen3.5-plus-02-15` (或具有優秀多模態視覺與推理能力的模型) 來處理圖片。

*   **核心套件**: `openai`, `Pillow` (處理圖片與 Base64 轉換), `argparse` (處理 CLI 參數), `python-dotenv` (選用，處理環境變數), `os`, `json`。
*   **環境變數**: 必須在系統中設定 `OPENROUTER_API_KEY` (安全考量，切勿寫死在程式碼中)。

## 2. CLI 流程設計 (Workflow)

一個完整的 CLI 批次處理指令應該長這樣：
`python translate_batch.py --input ./raw_manga --output ./translated_manga`

### 步驟流程圖：
1.  **初始化 (Initialization)**：解析 CLI 參數，建立輸出資料夾，從系統讀取 API Key 並實例化 OpenAI Client。
2.  **批次讀取 (Batch Loading)**：掃描 `--input` 資料夾下的所有圖片檔案 (如 `.png`, `.jpg`)。
3.  **圖片處理迴圈 (Image Processing Loop)**：
    *   讀取圖片並進行 Base64 編碼。
    *   組合 Prompt (包含 System Prompt、我們之前定義的 JSON Schema 規則與 `[0,1]` 相對座標要求)。
    *   **呼叫 API** (開啟 Reasoning 推理以增加翻譯準確度)。
    *   解析 API 回傳的 JSON (包含座標、翻譯文本、排版設定)。
    *   (可選) 呼叫本地背景擦除 (Inpainting) 與渲染 (Rendering) 函數將文字畫上圖片。
4.  **輸出儲存 (Output Saving)**：
    *   將 LLM 的 JSON 結果存入 `--output` 資料夾 (例如 `page01_data.json`)。
    *   若有執行繪圖，將最終圖片存入 `--output` 資料夾 (例如 `page01_translated.png`)。

---

## 3. Python 核心程式碼架構範例

核心程式碼範例：

```python
import os
import glob
import json
import base64
import argparse
from openai import OpenAI
# from dotenv import load_dotenv # 可選：讀取 .env 檔案

def encode_image_to_base64(image_path: str) -> str:
    """將圖片轉換為 Base64 字串以供 API 使用"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def call_llm_for_manga_page(client: OpenAI, base64_image: str) -> str:
    """呼叫多模態模型取得翻譯與排版 JSON"""
    
    # 這裡放入我們在 e2e-llm-workflow.md 中設計的強力 Prompt
    system_prompt = \"\"\"
    你是一個頂級的漫畫在地化翻譯專家與排版設計師。
    請找出圖片中所有文字，並回傳嚴格的 JSON。包含 box_id, original_text, translated_text, position (嚴格的 [0,1] 之間 3 到 4 位小數相對座標), 以及 render_config。
    \"\"\"

    # 使用 OpenRouter 特定的推理參數 (Extra Body)
    response = client.chat.completions.create(
        model="qwen/qvq-72b-preview", # 請替換為 OpenRouter 上支援多模態的強大模型
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": "請分析這頁漫畫並輸出指定的 JSON 格式。"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}
        ],
        response_format={"type": "json_object"}, # 強制模型回傳 JSON
        extra_body={"reasoning": {"enabled": True}}, # 開啟 OpenRouter 特有的推理能力
        temperature=0.3 # 降低溫度以獲取更穩定且規範的座標輸出
    )
    
    return response.choices[0].message.content

def process_directory(input_dir: str, output_dir: str):
    """處理整個資料夾的批次邏輯"""
    
    # 1. 建立輸出資料夾
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. 從系統環境變數獲取 API Key (OS GET)
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("請先設定環境變數 OPENROUTER_API_KEY！")
        
    # 初始化 OpenAI Client (指向 OpenRouter)
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    
    # 3. 取得所有圖片
    image_paths = glob.glob(os.path.join(input_dir, "*.jpg")) + glob.glob(os.path.join(input_dir, "*.png"))
    
    for img_path in image_paths:
        base_name = os.path.basename(img_path)
        print(f"[{base_name}] 開始處理...")
        
        try:
            # Step A: 讀圖與編碼
            encoded_img = encode_image_to_base64(img_path)
            
            # Step B: 呼叫 API
            print(f"[{base_name}] 呼叫 API 進行推理與翻譯...")
            json_result_str = call_llm_for_manga_page(client, encoded_img)
            
            # Step C: 解析回傳的 JSON
            result_data = json.loads(json_result_str)
            
            # --- 這裡可以插入將結果畫回圖片的渲染程式碼 (Renderer) ---
            # render_to_image(img_path, result_data, output_dir)
            
            # Step D: 儲存 JSON 結果到目的地資料夾
            output_json_path = os.path.join(output_dir, f"{os.path.splitext(base_name)[0]}.json")
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
                
            print(f"[{base_name}] 處理完成！結果已存至 {output_json_path}")
            
        except Exception as e:
            print(f"[{base_name}] 處理失敗：{e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI 批次漫畫翻譯工具")
    parser.add_argument("--input", "-i", type=str, required=True, help="輸入漫畫圖片的資料夾路徑")
    parser.add_argument("--output", "-o", type=str, required=True, help="輸出結果的資料夾路徑")
    
    args = parser.parse_args()
    process_directory(args.input, args.output)
```

## 4. 執行說明

在使用此 CLI 工具前，請在終端機 (Terminal / Command Prompt) 設定您的 API 金鑰：

**Windows (PowerShell):**
```powershell
$env:OPENROUTER_API_KEY="您的_API_KEY"
```

**運行批次腳本：**
```bash
python translate_batch.py -i C:\my_manga_raw -o C:\my_manga_translated
```

執行後，程式會自動將圖片交由具備 Reasoning 能力的模型分析，並將包含精確 `[0,1]` 座標與排版設定的 JSON 存入 `C:\my_manga_translated` 目錄。
