# 影片索引審核介面：使用說明

## 開啟

1. 在專案資料夾開啟 PowerShell，執行：
   ```
   D:\conda\envs\nbatrack\python.exe tools\index_server.py --out outputs\index\clip6
   ```
2. 用 Chrome 或 Edge 開啟 **http://127.0.0.1:8765**（CVAT 用 8080，兩者可以同時開）。
3. 用完在 PowerShell 視窗按 **Ctrl+C** 關閉。

這個介面只在本機運作。影片直接讀原始檔，不會另外複製或上傳。

## 畫面

**左側**：影片、目前時間、幀號、**比賽時鐘**（從記分板讀出），下面是時間軸，由上到下五列：

| 列 | 內容 |
|---|---|
| 畫面 | 綠色＝主轉播畫面，橘色＝特寫等其他畫面 |
| 非比賽 | 紅色，例如比賽時鐘停止的時間 |
| 候選回合 | 紫色＝湖人進攻，白色＝金塊進攻，灰色＝未知；橘框＝需要審核 |
| 人工 | 你自己新增的事件 |
| 事件 | ★＝得分，藍線＝進攻時鐘重置 |

在時間軸上點一下，就會跳到那個時間並選取該事件。

**右側**：
- **事件清單**：上方有快速篩選按鈕（候選回合、得分變化、需要審核、投籃、所有人工事件），也能依類型、狀態或文字搜尋。點一列就跳到該事件的開始時間。
- **編輯區**：
  - 開始/結束時間：可直接輸入，也可以按「= 目前時間」，或用 ±1 幀微調。
  - 其他欄位：類型、進攻方、標籤（用逗號分隔，例如 `pick_and_roll, made`）、審核狀態、備註。
  - 按鈕：
    - **播放此區間**：從開始播到結束。
    - **✓ 確認**、**✗ 否決**：改變審核狀態。自動事件不能刪除，只能否決。
    - **複製**：複製選取的事件。
    - **匯出片段**：輸出成 mp4，存到 `outputs\index\clip6\clips\`。
  - **新增事件**：先選類型（投籃、失誤、犯規等），再按「＋ 在目前時間新增事件」。

## 快捷鍵（游標不在輸入框時）

| 按鍵 | 功能 |
|---|---|
| 空白鍵 | 播放 / 暫停 |
| ← / → | 前後 1 幀 |
| Shift + ← / → | 前後 1 秒 |
| `[` / `]` | 把選取事件的開始 / 結束設為目前時間 |
| Ctrl+S | 存檔 |

## 存檔

- 存檔會寫入 `outputs\index\clip6\events_reviewed.json`，並把前一版自動備份到 `history\`。
- 自動產生的 `events_auto.json` 永遠不會被改動，所以重新跑索引不會蓋掉你的修正。
- 有未存檔的修改時，上方會顯示「未存檔」。
- 你確認（confirmed）或修改過（edited）的回合，之後會當作真值，用來計算邊界的精確率和召回率：
  ```
  D:\conda\envs\nbatrack\python.exe tools\index_eval.py --out outputs\index\clip6
  ```

## 重新產生索引（例如換一支影片）

```
D:\conda\envs\nbatrack\python.exe tools\index_video.py extract <影片> --scoreboard <scoreboard.json> --out outputs\index\<名稱>
D:\conda\envs\nbatrack\python.exe tools\index_video.py glyphs --out outputs\index\<名稱> --labels <glyph_labels.json>
D:\conda\envs\nbatrack\python.exe tools\index_video.py index --out outputs\index\<名稱> --scoreboard <scoreboard.json> --labels <glyph_labels.json>
```

同一家轉播商、同一季的記分板，`scoreboard.json` 和 `glyph_labels.json` 可以直接沿用。遇到新的記分板樣式時：
1. 先調整欄位位置（ROI）。
2. 對照 `glyph_clusters_sheet.png`，為每個字形群組填一次字元。
