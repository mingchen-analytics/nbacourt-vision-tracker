# 標註指南（CVAT）：Clip 2 與 Clip 5

目的：替系統建立**經過人工審核的真值**，用來客觀量測以下各項：

- 偵測、追蹤和 ID 一致性；
- 隊伍與裁判的分類；
- 腳底接觸點；
- 校正。

你不需要了解內部資料格式。照下面的步驟操作，最後匯出一個 zip 檔即可。

> 重要：系統提供的框只是**草稿（預標註）**，不是真值。每條軌跡你確認過之後，才能把它設成 `reviewed`。只要還有任何一條軌跡是 `unreviewed`，評估程式就會拒絕把結果當成真值。

---

## 0. 檔案在哪裡

| 檔案 | 位置 |
|---|---|
| 原始影片 | `data/raw/拓荒者 進攻 戰術 AI Ghost Flare Sharpe 擋拆助攻外線Camara.mp4`（clip2）<br>`data/raw/勇士 進攻 戰術 邊線 Double Gap Butler切入製造犯規.mp4`（clip5） |
| 標註套件 | `outputs/annotation/clip2/`、`outputs/annotation/clip5/` |
| 地標位置圖 | `docs/court_landmarks.png` |

每個套件資料夾裡有：

- `01_labels_for_cvat.json`：建立任務時要貼上的標籤設定。
- `02_preannotations_CVAT_video_1.1.zip`：系統的草稿框。
- `03_frames_to_annotate.txt`：要標腳底點和地標的幀號。
- `04_reference_frames/`：上述幀的截圖，左上角有幀號。
- `05_team_definition.txt`：A、B 隊各是哪一隊。兩支影片都是 **A = 拓荒者（黑衣）、B = 勇士（白衣）**。

## 1. 選擇 CVAT 環境（二選一）

| 選項 | 優點 | 缺點 |
|---|---|---|
| **A. app.cvat.ai（雲端，免費帳號）** | 不用安裝，馬上能用 | 影片會上傳到 CVAT 的雲端。這是 NBA 轉播畫面，是否可以上傳由你判斷。 |
| B. 本機自架 CVAT | 資料不離開電腦 | 需要安裝 WSL2 + Docker Desktop，加上 CVAT 映像檔，共數 GB，而 C 槽目前只剩約 18 GB。若選這條路，請先告訴我，我再提供安裝步驟。 |

兩者的操作介面相同，以下步驟都適用。

## 2. 建立任務（每支影片做一次）

1. 登入 CVAT，點 **Tasks → + → Create a new task**。
2. **Name**：輸入 `clip2`（或 `clip5`）。
3. 設定 **Labels**：
   - 點 **Raw** 分頁，把 `01_labels_for_cvat.json` 的全部內容貼上，覆蓋原本的內容。
   - 再點 **Done**。
   - 完成後應該看到 5 個標籤：player、referee、other_person、foot_contact、court_landmark。
4. **Select files → My computer**：選對應的**原始 mp4**。
5. 展開 **Advanced configuration**：
   - **Image quality**：設成 95 以上。
   - **Frame step**：留空或 1；**Start frame**：0；**Stop frame**：留空。**一定不能跳幀**，否則幀號會對不上。
6. 點 **Submit & Open**。等它處理完影片後，確認任務頁面顯示的幀數：
   - clip2 應為 **451**
   - clip5 應為 **332**

   數字不同的話請停下來告訴我。

## 3. 匯入草稿

1. 在任務頁面點 **Actions → Upload annotations**。
2. Format 選 **CVAT 1.1**。
3. 選擇 `02_preannotations_CVAT_video_1.1.zip`，確認上傳。

## 4. 審核和修正軌跡（主要工作）

點任務裡的 **Job** 進入標註畫面。右側 **Objects** 清單列出所有軌跡；建議一次只顯示一條來審核（點眼睛圖示隱藏其他軌跡）。

**要標哪些人：**

- **要標**：場上的球員和裁判，只要身體有任何部分出現在畫面裡就要標。
- **可選**：站在場邊地板上的板凳球員、教練、工作人員，標成 `other_person`。
- **不標**：看台上的觀眾。

**框怎麼畫：**

- 從頭頂到腳底，包住整個人。
- 身體部分被其他球員擋住時，依你的判斷畫出完整的身體範圍（包含被擋住的部分），並勾選 **Occluded**（快捷鍵 Q）。

**每條軌跡的檢查步驟：**

1. **播放或逐幀前進**（F 前進、D 後退），看框有沒有一直跟著同一個人。
2. **框不準時**：在那一幀直接拖曳框的位置或大小。CVAT 會自動建立關鍵幀，並重新內插前後的框。草稿每 5 幀有一個關鍵幀，大部分只需要微調。
3. **交錯後 ID 跳到另一個人身上**：
   - 在第一個錯誤的幀用 **Split**（剪刀工具）切開軌跡。
   - 再把錯的那一段併入正確的人；或者刪掉，重新為那個人建一條軌跡。
   - 草稿已經在系統懷疑換人的位置切成兩段了。如果你判斷其實沒換人，用 **Merge**（快捷鍵 M）把兩段合併。
   - **請特別檢查所有球員交錯、互相遮擋的時刻。**
4. **同一個人被拆成好幾條軌跡**：用 **Merge** 合併。
5. **系統漏掉的人**：
   - 左側選矩形工具，模式選 **Track**，標籤選 player 或 referee。
   - 在幾個關鍵幀畫框，中間 CVAT 會自動內插。
6. **人離開畫面或完全被擋住**：在他消失的第一幀按 **Outside**（快捷鍵 O）。重新出現時，再到那一幀關掉 Outside，並調整框的位置。
7. **標籤和屬性**：
   - 標籤選 `player`、`referee` 或 `other_person`。
   - player 的 `team` 依 `05_team_definition.txt` 選 A 或 B；真的無法判斷才選 unknown。
8. **整條軌跡確認無誤後**，把屬性 `review` 改成 **reviewed**。

隨時按 **Ctrl+S** 存檔。

## 5. 腳底接觸點（只標指定的幀）

幀號列在 `03_frames_to_annotate.txt` 的 FOOT 那一行（clip2 共 15 幀，clip5 共 11 幀）。

1. 在畫面上方的幀號欄輸入幀號，跳到該幀。
2. 左側選 **Points** 工具，標籤選 `foot_contact`，模式選 **Shape**（不要選 Track）。
3. 每個看得到腳的球員或裁判點**一個點**：
   - **兩腳都著地**：點在兩腳與地板接觸點的中間。
   - **只有一腳著地**：點在那隻腳的著地點。
   - **兩腳都離地（跳起來）**：點在較低那隻腳的正下方地板，`state` 選 **airborne**。
   - **腳被擋住看不到**：不要點。
4. 盡量點準，誤差最好在幾個像素以內，可以放大畫面再點。

> 這些點刻意**沒有預先填入**。如果用系統的估計當草稿，會讓評估結果偏向系統自己的答案。

## 6. 球場地標（只標指定的幀）

幀號列在 `03_frames_to_annotate.txt` 的 LANDMARK 那一行（每支影片 4 幀）。

1. 選 **Points** 工具，標籤選 `court_landmark`，模式選 **Shape**。
2. 只在**清楚可見的線條交點**上點，然後在 `landmark` 下拉選單選對應的名稱。名稱和位置請對照 `docs/court_landmarks.png`。
3. 最可靠、優先標的地標：
   - 禁區和罰球線的兩個交點：`*_lane_ft_far`、`*_lane_ft_near`
   - 中線和邊線的交點：`mid_far`、`mid_near`
   - 底角三分線和底線的交點
4. **不要標：**
   - 被人擋住的交點。
   - **看不出交點的位置。** 例如勇士主場底線外的區域也是黃色，禁區和底線的交角根本看不出來。
   - 弧線的「頂點」：`*_ft_circle_top`、`*_3pt_top`。它們沿著弧線方向的位置很難判斷，除非你非常確定。
5. 每幀能標到 4–8 個點最理想；只有 2–3 個清楚的點也可以。

## 7. 匯出並交給系統

1. 在任務頁面點 **Actions → Export task dataset**。
2. Format 選 **CVAT for video 1.1**，**不要**勾選 Save images。
3. 下載 zip 檔，存成：
   - `data/annotations/clip2/gt_cvat_export.zip`
   - `data/annotations/clip5/gt_cvat_export.zip`
4. 告訴我，或自己執行：
   ```
   D:\conda\envs\nbatrack\python.exe tools\evaluate_against_gt.py --gt data\annotations\clip2\gt_cvat_export.zip --runs outputs\real\clip2_ptz --out outputs\eval\clip2
   ```

## 預估時間

| 項目 | clip2（15 秒） | clip5（11 秒） |
|---|---|---|
| 建立任務並匯入 | 10 分鐘 | 10 分鐘 |
| 審核和修正軌跡 | 2–3 小時 | 1.5–2 小時 |
| 腳底點（每幀約 10 人） | 30–40 分鐘 | 25–30 分鐘 |
| 地標（4 幀） | 15 分鐘 | 15 分鐘 |

可以分好幾次做，CVAT 會保存進度。建議**先完成 clip2**。
