---
description: [Manga-Image-Translator 渲染與排版 (Rendering) 程式碼架構與原理]
---

# Manga-Image-Translator 渲染與排版 (Rendering) SKILL

## 1. 模組架構與檔案索引

渲染模組位於 `manga_translator/rendering/` 目錄下。主要透過 OpenCV (`cv2`)、NumPy (`np`) 與 FreeType (`freetype`) 進行像素級別的文字繪製。

### 核心檔案索引
*   **`manga_translator/rendering/__init__.py`**
    *   **角色**：渲染流程的總控與分發中心 (Dispatcher)。
    *   **核心函式**：`dispatch()`
        *   接收前一步驟 (Inpainting) 處理完的無字背景圖、原文 Bounding Box (對話框座標)、翻譯後的文字。
        *   根據設定決定要呼叫哪一套渲染器 (例如預設的 `text_render.py`，或是針對適配英文字母框的 `text_render_eng.py`)。
*   **`manga_translator/rendering/text_render.py`** (★ 最重要)
    *   **角色**：預設的排版與繪製引擎，包含完整的字元寬高計算、斷詞、動態換行與 OpenCV 繪製邏輯。
    *   **核心函式**：
        1.  **`calc_horizontal` / `calc_vertical`**：排版演算法核心。負責計算給定文字在特定長寬限制下，應該如何斷行、每行有哪些字、以及最終文字框的長寬。
        2.  **`put_char_horizontal` / `put_char_vertical`**：單字元渲染核心。負責呼叫 FreeType 取得字形 Bitmap，並透過 NumPy 切片將字元與「外框描邊 (Stroke)」寫入畫布矩陣。
        3.  **`put_text_horizontal` / `put_text_vertical`**：行渲染器。負責根據 `calc` 算出的座標，將一行行文字透過 `put_char` 畫到整張透明畫布上，最後呼叫 `add_color` 上色並返回去背的文字圖層。
*   **`manga_translator/rendering/text_render_eng.py`**
    *   **角色**：針對英文字體與美漫對話框 (`--manga2eng`) 最佳化的渲染器。
    *   **特點**：西文漫畫的對話框通常是橫橢圓形，此腳本包含更複雜的「找尋最大內接矩形/多邊形」演算法，盡可能讓文字貼合橢圓邊緣，而非傳統日漫的矩形排版。
*   **`manga_translator/rendering/ballon_extractor.py`**
    *   **角色**：對話框形狀特徵提取。用於輔助決定文字邊界。

---

## 2. 核心排版演算法解析 (`text_render.py` - `calc_horizontal`)

### 2.1 斷詞與換行策略
*   為了避免將英文單字從中間切斷，程式會先用 Regex `\s+` 將字串切成 Word List。
*   引入 `hyphen` 庫：如果單字超過行寬限制，會利用音節字典尋找合適的斷點，並加上 `-` (連字號)。
*   中文與日文則以單字元為單位直接評估。

### 2.2 彈性空間擴展 (Elastic Box)
當文字預估面積超過 Bounding Box 時：
```python
# text_render.py 內的部分邏輯概念：
expected_size = sum(word_widths) + 空白與連字號寬度
max_size = max_width * max_lines
if max_size < expected_size:
    # 如果原本的框塞不下，不直接縮小字體，而是按比例放大框的長寬限制
    multiplier = np.sqrt(expected_size / max_size)
    max_width *= max(multiplier, 1.05)
    max_height *= multiplier
```
這解釋了為何自動排版有時會溢出原本的氣泡框。

### 2.3 貪婪換行法 (Greedy Line Breaking)
計算好彈性空間後，程式採用貪婪演算法，逐字累加寬度，一旦預視到下一個字會超過 `max_width`，就強制截斷並開啟新的一行。

---

## 3. 像素渲染與描邊特效 (`text_render.py` - `put_char_horizontal`)

1.  **取得字形 Bitmap**
    *   呼叫 `FreeType` 載入 `.ttf` / `.ttc`，取得該字元的二維灰階矩陣 (`bitmap.buffer`)。
2.  **描邊生成 (Stroke Generation)**
    *   呼叫 `freetype.Stroker()`。
    *   半徑公式：`stroke_radius = 64 * max(int(0.07 * font_size), 1)` (描邊粗細約為字體的 7%)。
    *   樣式：設定為 `FT_STROKER_LINEJOIN_ROUND` (圓角連接)，避免描邊出現尖銳毛刺。
3.  **多圖層寫入**
    *   建立 `canvas_text` (放黑字) 與 `canvas_border` (放白邊)。
    *   利用 `NumPy` 切片 (例如 `canvas_text[y:y+h, x:x+w] = bitmap_char`) 將字形寫入矩陣。
    *   使用 OpenCV `cv2.add` 確保描邊圖層在字距過近時能平滑融合，不會互相覆蓋出黑洞。
4.  **結合**
    *   最後在 `put_text_horizontal` 函數中，呼叫 `cv2.boundingRect` 切齊透明邊界，並疊加前景與背景色，回傳最終的 ARGB 圖層給主引擎。
