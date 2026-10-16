# NBA Broadcast Video-to-Tracking — Phase 1 研究與架構提案

> 狀態：**已核准（2026-10-10），附下列修訂**。本文件中所有效能數字若標示「目標」即為尚未量測的驗收門檻，不是實測結果。
> 日期：2026-10-10

## 核准時的修訂決策（使用者）
1. 不等真實影片：先做幾何與合成測試。
2. **暫不訓練**球場地標模型。優先做半自動校正，並調查可下載的 pretrained 權重（調查結果見 `docs/LICENSES.md`）。
3. 主實作不依賴 AGPL 套件。Ultralytics 只能作為獨立的研究基線；所有授權都要先記錄。
4. 環境、權重、資料一律放 D 槽。
5. **校正介面必須同時支援 PTZ 與一般 homography，並以實證比較。** 固定光學中心的假設要經過測試，不能直接預設（實作在 `calib/models.py`、`calib/fit.py`、`eval/calib_compare.py`）。
6. 0.5 m 只是實驗目標。偵測、地面點、校正、最終座標四種誤差要分開量測；不得用同一個校正模型投影出來的點去宣稱公尺誤差。
7. 地面接觸點先用 bbox 底部作為基線，pose 方法另外評估，並先確認 checkpoint 確實有腳跟和腳趾關鍵點。
8. CPU 為基準；DirectML 要實測，不能預設會比較快。

---

## 0. 環境檢查結果（實測）

| 項目 | 結果 | 影響 |
|---|---|---|
| CPU | AMD Ryzen 7 7840HS，8 核 / 16 執行緒 | 推論主力 |
| GPU | AMD Radeon 780M 內顯（gfx1103），**無 NVIDIA GPU**，`nvidia-smi` 不存在 | 無 CUDA。AMD Windows ROCm/PyTorch 支援清單不含 780M → 不能訓練大型模型 |
| NPU | 「NPU Compute Accelerator Device」存在（Ryzen AI / XDNA） | 生態不成熟，暫不依賴 |
| RAM | 31.2 GB | 足夠 |
| 磁碟 | C: 22 GB 可用（偏少）；D: 323 GB 可用 | **環境、模型、影片都放 D 槽** |
| Python | miniconda base = Python 3.14（太新，部分套件可能無 wheel） | 建立獨立 conda env，Python 3.11 |
| FFmpeg | 8.1 full build，支援 d3d11va / amf 硬體解碼 | 可用於解碼與精確時間戳 |
| 專案目錄 | 空目錄，非 git repo | 從零建立 |

**結論**：本機適合「離線、CPU / DirectML 推論」，不適合訓練。任何模型微調（court keypoint、ball detector）需要外部 GPU（例如 Google Colab），屆時會先與你確認成本。

---

## 1. 可行性評估

### 1.1 可行的部分
- **主轉播機位（sideline main camera）的球場校正是可行的**。NBA 球場尺寸標準化、線條清楚，且主機位物理上近似「固定位置的 PTZ 攝影機」（位置固定，只有 pan / tilt / zoom 改變）。這是比足球更有利的條件。
- 球員偵測與追蹤已有成熟基線；SportsMOT 基準上 Deep-EIoU 約 77 HOTA、BoT-SORT 約 69、ByteTrack 約 64（論文報告值，非本專案實測）。
- 「影像 → 地面座標」對於**雙腳著地**的球員，幾何上是嚴格成立的（地板是平面，平面到影像是 homography）。

### 1.2 根本限制（無法靠工程完全消除）
| 限制 | 量級估計（推算，待驗證） |
|---|---|
| **遠側邊線解析度低**：1080p 下遠側每公尺約只有數十像素，1 px 誤差可能對應數公分到十幾公分 | 遠側誤差顯著大於近側 |
| **球員跳躍**：腳離地時，把腳投影到地面會沿視線方向往遠處偏移。相機俯角約 20–30° 時，0.5 m 跳躍高度 ≈ 1 m 水平誤差 | 需偵測並標記 airborne，不能平滑掩蓋 |
| **腳部遮擋**（禁區擠在一起） | 退回 bbox 底部中心並放大不確定度 |
| **只看到半場 / 特寫** 時地標不足 | PTZ 模型可降低所需地標數；仍不足則標記 calibration=failed |
| **球的 3D 位置**：單一視角下空中的球深度不可觀測 | 預設 3D = unknown，只存影像座標 |
| **跨鏡頭切換的身分一致性**：沒有球衣號碼辨識時無法保證 | 每個 camera segment 內 track_id 獨立，不宣稱跨段一致 |

### 1.3 合理的精度目標（MVP 驗收門檻，非承諾）
- 主機位、校正品質 good 的影格：球員地面位置**中位誤差 ≤ 0.5 m**（相對人工標註真值）。
- 文獻與業界經驗顯示，良好校正下達到 0.3–0.5 m 是合理的；但必須以本專案量測為準。

---

## 2. 開源方案調研與比較

### 2.1 球場校正 / 場地註冊（最高優先）
| 方案 | 類型 | 籃球適用 | 可用性 / 授權 | 評估 |
|---|---|---|---|---|
| Roboflow `basketball-court-detection-2` 資料集 + RF-DETR Keypoint / YOLO-pose | 33 個球場地標關鍵點 → homography | 專為 NBA 轉播 | 資料集在 Roboflow Universe；**未找到可直接下載的已訓練權重**；資料集授權需在下載頁確認 | 最直接可用的資料來源，但需自行訓練 |
| KaliCalib (MMSports 2022) | keypoint 熱圖 → homography | 籃球 | 論文；程式碼可用性未確認 | 方法參考 |
| PnLCalib / No-Bells-Just-Whistles | 點 + 線聯合非線性最佳化，完整 3D 相機 | 足球（可改寫成籃球場模型） | 官方 PyTorch 程式碼 | **「點+線精修」與「3D 相機模型」的思路直接採用** |
| TVCalib | 線段對應直接最佳化相機參數 | 足球 | 開源 | 精修步驟參考 |
| DeepSportRadar calibration dataset | 籃球校正資料 | 籃球（但非 NBA 轉播風格） | CC BY-NC-ND / NC-SA（來源說法不一） | 只可研究用，可做外部驗證 |
| roboflow/sports | 工具函式庫 | main 分支目前**只有足球**設定，沒有籃球模組 | MIT | 可參考其 ViewTransformer 寫法，不直接依賴 |

### 2.2 球員偵測
| 方案 | 授權 | 評估 |
|---|---|---|
| **RF-DETR (Nano–Large)** | **Apache 2.0**（XL/2XL 為 PML 1.0，不用） | 官方 COCO AP 高於同級 YOLO11（廠商數據）；可匯出 ONNX → 本機 DirectML/CPU |
| Ultralytics YOLO11 / YOLO-pose | **AGPL-3.0** | 生態完整、CPU 速度快；授權具傳染性。建議僅作「比較基線」，需你同意 |
| COCO `person` 預訓練 | — | 會同時抓到裁判、觀眾、板凳 → 需「場內遮罩 + 隊伍/裁判分類」過濾 |

### 2.3 多目標追蹤
| 方案 | 授權 | SportsMOT HOTA（論文值） | 評估 |
|---|---|---|---|
| ByteTrack（supervision 實作） | MIT | ~64 | 穩定基線，第一版使用 |
| BoT-SORT（含相機運動補償） | boxmot 為 AGPL-3.0 | ~69 | CMC 思路自行實作 |
| Deep-EIoU（擴張 IoU + ReID） | 開源 | ~77 | Phase 3 改進方向 |
| Roboflow 的 SAM2 追蹤 | Apache 2.0 | — | T4 上僅 1–2 FPS，CPU 不可行，**不採用** |

**本專案特有優勢**：校正成功後可以在「球場座標（公尺）」中做關聯，相機平移/縮放的影響被消除，運動模型（球員速度 ≤ ~10 m/s）有物理意義。計畫採用「影像 IoU + 球場距離門檻 + 外觀特徵」的混合關聯。

### 2.4 地面接觸點（ground-contact）
| 方案 | 授權 | 評估 |
|---|---|---|
| **RTMPose Halpe26（rtmlib，ONNX）** | rtmlib/MMPose 授權需確認（預期 Apache 2.0） | 有腳跟、大腳趾、小腳趾關鍵點；只依賴 numpy/opencv/onnxruntime，**輕量、CPU 可行** |
| ViTPose + feet | Apache 2.0 | 較重 |
| bbox 底部中心 | — | 只作備援，並標記較大不確定度 |

（注意：專案範圍排除「關節/骨架追蹤」。這裡只用腳部關鍵點作為**地面接觸估計器**，不輸出骨架資料。）

### 2.5 隊伍區分
- Roboflow 方案用 SigLIP + UMAP + KMeans，CPU 上偏重。
- **第一版**：軀幹區域 HSV/Lab 色彩直方圖 + KMeans（k=3：隊 A、隊 B、裁判），以 track 為單位做時間投票。可量測後再決定是否升級為深度外觀特徵。

### 2.6 其他元件
| 用途 | 選擇 | 理由 |
|---|---|---|
| 影片解碼與時間戳 | **PyAV**（FFmpeg 綁定，BSD） | 直接取 PTS；OpenCV 的 `CAP_PROP_POS_MSEC` 對 VFR 影片不可靠 |
| 鏡頭切換偵測 | PySceneDetect（BSD-3）+ 自製規則 | 輕量 |
| 幾何 | OpenCV + SciPy（least_squares） | 標準 |
| 資料輸出 | pyarrow（Parquet）+ CSV + JSON；DuckDB 選用 | SQL 可直接查 Parquet |
| 標註 | CVAT（MIT） | 支援 MOT 格式、內插 |
| 評估 | TrackEval（HOTA/IDF1/IDSW） | 社群標準 |

---

## 3. 架構圖

```
                         ┌──────────────────────────────────────────┐
  MP4 ──► VideoReader ──►│ Frame(frame_index, pts, timestamp_sec,   │
       (PyAV, PTS)       │       image)                             │
                         └───────────────┬──────────────────────────┘
                                         │
                         ┌───────────────▼──────────────┐
                         │ Segmenter                    │  鏡頭切換偵測
                         │  shot boundary + view class  │  → camera_segment_id, view_type
                         └───────┬───────────────┬──────┘
                    unsupported  │               │ main_broadcast
                 (標記並跳過)◄───┘               │
        ┌────────────────────────────────────────┼─────────────────────────────┐
        │                                        │                             │
┌───────▼─────────┐               ┌──────────────▼───────────────┐   ┌─────────▼────────┐
│ CourtCalibrator │               │ PlayerDetector (RF-DETR ONNX)│   │ BallDetector     │
│ 1 landmark det. │               │  → bbox, conf                │   │ HoopDetector     │
│ 2 line det.     │               └──────────────┬───────────────┘   │  → image-space   │
│ 3 H (RANSAC)    │                              │                   └─────────┬────────┘
│ 4 PTZ 相機模型  │                 ┌────────────▼────────────┐                 │
│ 5 幀間運動傳遞  │                 │ GroundContactEstimator  │                 │
│ 6 時間平滑(RTS) │                 │ (RTMPose 腳部關鍵點)     │                 │
│ 7 品質旗標      │                 └────────────┬────────────┘                 │
│ ▲ 手動校正備援  │                              │ image_x, image_y, σ_px       │
└───────┬─────────┘                              │                             │
        │ H_t, camera_t, quality_t   ┌───────────▼────────────┐                │
        ├───────────────────────────►│ CourtProjector         │                │
        │                            │ 影像→球場(m) + 誤差傳遞 │                │
        │                            │ + 場內遮罩過濾          │                │
        │                            └───────────┬────────────┘                │
        │                            ┌───────────▼────────────┐                │
        │                            │ Tracker                │                │
        │                            │ IoU + 球場距離 + 外觀   │                │
        │                            │ TeamClassifier          │                │
        │                            └───────────┬────────────┘                │
        │                            ┌───────────▼────────────┐                │
        │                            │ TemporalFilter         │                │
        │                            │ raw 保留 / filtered 另存│                │
        │                            │ airborne & 物理合理性   │                │
        │                            └───────────┬────────────┘                │
        └───────────────┬────────────────────────┴─────────────────────────────┘
                        ▼
              ┌──────────────────────┐     ┌──────────────────────────────┐
              │ Exporter             │────►│ Parquet / CSV / JSON          │
              │ (schema v1, 明確 null)│     │ players, ball, calibration,   │
              └──────────┬───────────┘     │ landmarks, segments, video    │
                         │                 └──────────────────────────────┘
              ┌──────────▼───────────┐     ┌──────────────────────────────┐
              │ Visualizer           │────►│ side-by-side MP4（左原片疊圖，│
              └──────────────────────┘     │ 右俯視球場）                   │
                                           └──────────────────────────────┘
              ┌──────────────────────┐
              │ Evaluator            │  偵測 P/R、HOTA/IDF1/IDSW、重投影誤差、
              │ (對照人工標註)        │  位置誤差(m)、覆蓋率、FPS
              └──────────────────────┘
```

每個模組是獨立的 Python 介面（輸入/輸出為 dataclass），可單獨測試、替換模型，也方便日後串流化（逐幀處理，只有 RTS 平滑需要離線反向傳遞，串流時改用 Kalman 前向濾波）。

---

## 4. 專案目錄結構

```
NBA_Track_app/
├── pyproject.toml              # 套件與依賴
├── environment.yml             # conda env（Python 3.11）
├── README.md                   # 安裝、處理新影片的步驟
├── configs/
│   └── default.yaml            # 所有門檻、模型路徑、參數
├── docs/
│   ├── PHASE1_ARCHITECTURE.md  # 本文件
│   ├── COORDINATE_SYSTEM.md
│   ├── DATA_SCHEMA.md
│   └── ANNOTATION_GUIDE.md
├── src/nbatrack/
│   ├── io/video.py             # PyAV 讀取、PTS→timestamp
│   ├── court/
│   │   ├── geometry.py         # NBA 球場模型（公尺）、地標、線段、籃框 3D
│   │   └── render.py           # 俯視圖繪製
│   ├── calib/
│   │   ├── landmarks.py        # 地標偵測（模型推論）
│   │   ├── lines.py            # 白線/邊線偵測
│   │   ├── homography.py       # DLT + RANSAC + LM 精修
│   │   ├── camera_model.py     # PTZ 相機模型、H 分解
│   │   ├── propagate.py        # 幀間運動（旋轉相機的 homography）
│   │   ├── smoothing.py        # Kalman / RTS 平滑
│   │   ├── quality.py          # 品質分數與旗標
│   │   └── manual.py           # 手動點選校正工具
│   ├── segments/               # 鏡頭切換、機位分類
│   ├── detect/                 # players.py, ball.py, hoop.py
│   ├── pose/ground_contact.py  # 腳部關鍵點 → 地面接觸點
│   ├── track/                  # tracker.py, appearance.py, teams.py
│   ├── project/projector.py    # 影像→球場、誤差傳遞、場內過濾
│   ├── filter/temporal.py      # 時間濾波、airborne 偵測、物理檢查
│   ├── export/                 # schema.py, writers.py
│   ├── viz/                    # overlay.py, side_by_side.py
│   ├── eval/                   # detection, tracking, calibration, position
│   ├── pipeline.py             # 串接所有模組
│   └── cli.py                  # `nbatrack process / calibrate / eval / viz`
├── tools/                      # 標註、除錯小工具
├── tests/
│   ├── unit/                   # 幾何、投影、schema
│   └── integration/            # 合成球場影片端到端測試
├── models/                     # 權重（gitignore）
├── data/                       # 影片、標註（gitignore）
└── outputs/                    # 結果（gitignore，附少量樣本）
```

---

## 5. 球場座標系統（明確定義）

- **單位**：公尺（m）、秒（s）。原始英制尺寸換算：1 ft = 0.3048 m。
- **原點**：中場圓心，在地板平面上。
- **x 軸**：沿球場長邊（邊線方向），**正向指向主轉播機位畫面右側的籃框**。
- **y 軸**：沿短邊（底線方向），**正向指向遠離主機位的遠側邊線**。
- **z 軸**：垂直向上（右手座標系）。地板 z = 0。
- 俯視圖：x 向右、y 向上 → 主機位側邊線在下方，與轉播畫面方向一致。
- 「左/右籃框」是相對主機位定義，不是相對球隊（半場會換邊）。

| 元素 | 座標 / 尺寸 |
|---|---|
| 球場邊界 | x ∈ [−14.326, 14.326]，y ∈ [−7.620, 7.620]（94 ft × 50 ft） |
| 籃框中心 | (±12.725, 0, 3.048)：距底線 5.25 ft，高 10 ft |
| 籃板面 | 距底線 4 ft（x = ±13.107），寬 6 ft |
| 禁區（paint） | 寬 16 ft（y ∈ [−2.438, 2.438]），自底線延伸 19 ft（罰球線 x = ±8.534） |
| 罰球圈半徑 | 6 ft = 1.829 m |
| 中圈半徑 | 6 ft = 1.829 m |
| 三分線 | 弧半徑 23.75 ft = 7.239 m（以籃框中心為圓心）；底角直線距籃框中心 22 ft（|y| = 6.706），自底線延伸 14 ft |
| 合理衝撞區 | 半徑 4 ft = 1.219 m |

注意：NBA 線寬 2 in（≈5 cm），尺寸量測基準（線內緣/外緣）在 `geometry.py` 中逐條註明，這是 ±5 cm 等級的系統誤差來源。

---

## 6. 相機校正方法論

### 6.1 核心觀點：用「PTZ 相機模型」而不是「每幀獨立 homography」
主轉播機位物理上是**固定位置**、只會 pan / tilt / zoom 的攝影機。因此：

- **每幀獨立 homography**：8 自由度，需要 ≥4 個非共線地標。只看到半場時常不穩定，幀間會抖動。
- **PTZ 模型**：相機中心 **C（整段甚至整場共用）**、每幀只有 pan、tilt、焦距 f（roll 近似固定）≈ **3 自由度**。2 個地標 + 線條即可解，且在物理上受約束。
- 額外好處：PTZ 模型是真正的 3D 相機，可以把**籃框（z = 3.048 m，不在地板平面上）** 投影回影像。若投影的籃框與偵測到的籃框吻合，就是一個**獨立於地板線條的校正正確性檢驗** —— 單純 homography 做不到。這也是日後球的 3D 估計的基礎。

### 6.2 流程
1. **地標偵測**：33 點球場關鍵點模型（需以 Roboflow 資料集微調）。MVP 階段在模型就緒前，用**手動關鍵幀校正**替代。
2. **線條偵測**：地板白線 / 邊線遮罩（顏色 + 邊緣 + 方向），遮蔽球員 bbox 與固定的記分板圖層。
3. **初始 homography**：DLT + RANSAC（≥4 點），再以 Levenberg–Marquardt 同時最小化「地標重投影誤差 + 投影球場線到偵測線的距離場殘差」（PnLCalib / TVCalib 的點+線思路）。
4. **分解為相機參數**：假設主點在影像中心、方形像素、零 skew，由 H 求 f、R、C。
5. **共用相機中心**：在同一 segment 的多個高品質關鍵幀上以 bundle adjustment 求 C（與 roll、徑向畸變 k1）。
6. **逐幀求解**：固定 C，求 (pan, tilt, f)。初值來自前一幀 + **幀間運動**：純旋轉相機的幀間影像變換本身就是 homography `K₂ R K₁⁻¹`，**不只地板，連觀眾席、看台的特徵點都可用**（遮掉球員即可），大幅提升穩定性。
7. **時間平滑**：對 (pan, tilt, f) 做 Kalman 濾波 + RTS 平滑（離線雙向）。**原始逐幀估計與平滑結果都保存**；殘差過大的幀標記為離群，不讓平滑掩蓋。
8. **品質旗標**：
   - `reproj_rmse_px`（地標重投影 RMSE）
   - `line_alignment`（投影球場線與偵測白線的 chamfer 距離）
   - `n_inliers`、相機參數物理合理性（C 高度、f 範圍）、與前後幀的一致性
   - 綜合為 `good / fair / poor / failed`；`poor` 以下**不輸出球場座標**（court_x/y = null，position_status = unknown）。
9. **非 PTZ 機位**：若 C 估計值與主機位 C 差異大 → 判定為不同機位 → 該 segment 退回自由 homography 並標記，或直接標記 unsupported。

### 6.3 手動 / 半自動備援
- `nbatrack calibrate --frame N`：OpenCV 視窗中點選 ≥4 個已知地標（有清單提示），立即顯示投影球場線疊圖供確認。
- 手動關鍵幀 → 步驟 6 的幀間傳遞 → 步驟 3 的線條精修自動跟上。
- 這也是 **MVP 不需要先訓練模型就能跑通的關鍵**。

### 6.4 不確定度傳遞
- 從 LM 最佳化的 Jacobian 估計相機參數共變異，加上腳部像素不確定度 σ_px，一階傳遞到球場座標 → 每筆位置輸出 `court_sigma_m`。遠側、腳部被遮擋時自然變大。

---

## 7. 資料 Schema 提案（v1）

通用規則：時間以 PTS 換算的 `timestamp_sec`（float64，秒）+ `frame_index`（int，解碼順序 0 起算）；像素座標以**原始解析度**、左上為原點；缺值一律 null（Parquet null / CSV 空字串），不用 0 或 −1 代替。

### 7.1 `videos`
`video_id, source_path, sha256, width, height, fps_nominal, time_base, duration_sec, n_frames, codec, pipeline_version, config_hash, created_at`

### 7.2 `camera_segments`
`video_id, camera_segment_id, start_frame, end_frame, start_sec, end_sec, view_type (main_broadcast|other|unknown), supported (bool), boundary_method, boundary_confidence`

### 7.3 `calibration`（每幀一列）
| 欄位 | 說明 |
|---|---|
| video_id, frame_index, timestamp_sec, camera_segment_id | |
| method | keypoint / propagated / manual / smoothed / none |
| H_img_to_court | 9 floats（row-major，影像 px → 球場 m） |
| H_court_to_img | 9 floats |
| focal_px, cx, cy, k1 | 內參 |
| pan_deg, tilt_deg, roll_deg | 外參（旋轉） |
| cam_x_m, cam_y_m, cam_z_m | 相機中心 |
| n_landmarks, n_inliers | |
| reproj_rmse_px, line_alignment_px | |
| quality | good / fair / poor / failed |
| quality_score | 0–1 |

### 7.4 `court_landmarks`（靜態參考表）
`landmark_id, name, court_x_m, court_y_m, court_z_m, description`（含籃框）

### 7.5 `landmark_observations`
`video_id, frame_index, landmark_id, image_x, image_y, confidence, is_inlier, reproj_error_px, source (model|manual)`

### 7.6 `player_observations`（每幀每位球員一列）
| 欄位 | 說明 |
|---|---|
| video_id, timestamp_sec, frame_index, camera_segment_id | |
| track_id | 字串，例如 `s003_t017`（segment 內唯一，**不宣稱跨段一致**） |
| team_id, team_confidence | A / B / null |
| role | player / referee / other |
| bbox_x1, bbox_y1, bbox_x2, bbox_y2 | px |
| det_confidence, assoc_confidence | |
| ground_method | feet_keypoints / bbox_bottom / none |
| image_x, image_y | 觀測到的地面接觸點 px |
| court_x_m_raw, court_y_m_raw | 直接投影（未平滑） |
| court_x_m, court_y_m | 時間濾波後 |
| court_sigma_m | 位置不確定度（1σ） |
| position_status | `observed_projected` / `estimated_temporal` / `unknown` |
| flags | 位元/字串：possibly_airborne, feet_occluded, off_court, speed_violation |
| calibration_quality | 取自 calibration 表 |
| is_visible, is_on_court | bool |

### 7.7 `ball_observations`
`video_id, timestamp_sec, frame_index, camera_segment_id, ball_track_id, bbox_*, image_x, image_y, det_confidence, is_visible, position_3d_status (unknown|trajectory_fit|floor_contact), ball_x_m, ball_y_m, ball_z_m (預設 null), ball_sigma_m`

**空中的球絕不直接投影到地板當作位置**；MVP 中 3D 一律 `unknown`。

### 7.8 `hoop_observations`
`video_id, frame_index, hoop_side (left|right), image_x, image_y, bbox_*, confidence, predicted_image_x, predicted_image_y, residual_px`（predicted 由相機模型投影 3D 籃框，用於驗證）

### 7.9 匯出
- Parquet（主格式，每表一檔，含 schema version metadata）、CSV（相同欄位）、`run_metadata.json`（設定、版本、模型雜湊、處理時間、FPS）。
- 可直接用 `pandas.read_parquet` 或 DuckDB `SELECT * FROM 'outputs/x/player_observations.parquet'` 查詢。

---

## 8. MVP（Phase 2）實作計畫

### 8.1 里程碑
| # | 內容 | 驗證方式 |
|---|---|---|
| M0 | conda env（D 槽）、專案骨架、CI 用 pytest | `pytest` 通過 |
| M1 | `court/geometry.py` + 俯視圖繪製 + 座標文件 | 單元測試：尺寸、對稱性、地標座標 |
| M2 | **合成測試平台**：給定已知 PTZ 參數，渲染球場線影像 → 驗證 homography / PTZ 解算可還原參數（誤差 < 0.1 px） | 這是幾何正確性的硬性單元測試 |
| M3 | VideoReader（PTS）+ 鏡頭切換偵測 | 與 `ffprobe` 逐幀時間戳比對一致 |
| M4 | 手動校正工具 + 幀間傳遞 + 線條精修 + 品質旗標 | 保留地標重投影誤差（held-out） |
| M5 | 球員偵測（RF-DETR ONNX）+ 場內過濾 + 速度基準 | 在標註幀上量 P/R、FPS |
| M6 | 腳部關鍵點 → 地面點 → 球場投影 + 不確定度 | 對照人工腳點 |
| M7 | ByteTrack 基線 + 球場距離關聯 + 隊伍分類 | HOTA / IDF1 / IDSW |
| M8 | 時間濾波（保留 raw）+ airborne / 速度旗標 | 物理合理性違規率 |
| M9 | 匯出（Parquet/CSV/JSON）+ side-by-side 視覺化 | schema 測試、人工目視 |
| M10 | 評估報告（全部實測數字） | `nbatrack eval` 產出報告 |
| 平行 | 在 Colab 上微調 33 點球場地標模型（需你同意） | 地標偵測 PCK / 重投影誤差 |

球的偵測：MVP 只用 COCO `sports ball` 或現成模型做影像座標，並誠實報告召回率；專用球偵測器放 Phase 3。

### 8.2 MVP 驗收標準（目標值，於 30–60 秒主機位片段上實測）
| 指標 | 目標 |
|---|---|
| 端到端 CLI 可重現執行 | 同設定兩次執行輸出一致 |
| 時間戳 | 與 ffprobe PTS 完全一致 |
| 校正：held-out 地標重投影誤差（1080p） | 中位數 ≤ 5 px |
| 主機位幀中 calibration ≥ fair 的比例 | ≥ 90% |
| 場上球員偵測（過濾後） | Precision ≥ 0.90、Recall ≥ 0.90 |
| 追蹤 | IDF1 ≥ 0.70，回報 IDSW 數 |
| 球場位置誤差（對照人工真值） | 中位 ≤ 0.5 m，並分近側/遠側回報 |
| 可見球員中有效球場座標比例 | ≥ 85% |
| 處理速度 | 實測並回報（預期 CPU 上數 FPS，不設門檻） |

### 8.3 時間估計
- 我的開發時間：M0–M10 約需 **多個工作階段、合計約 6–10 小時代理工作時間**（取決於測試片段難度與除錯量）。
- 你的人工時間：標註驗證資料約 **3–5 小時**（見 §9.2）。
- 處理速度推算（未實測）：CPU 上整體約 2–6 FPS → 60 秒 30fps 片段約 5–15 分鐘；整場比賽在本機可能需十幾到數十小時，Phase 4 再評估降採樣或雲端 GPU。

---

## 9. 風險與客觀驗證策略

### 9.1 主要風險（依嚴重度）
| # | 風險 | 緩解 |
|---|---|---|
| 1 | **沒有現成可下載的籃球場地標權重**，需訓練；本機無法訓練 | MVP 先用手動關鍵幀 + 傳遞；平行在 Colab 微調 |
| 2 | 僅 CPU，速度慢 | ONNX + DirectML/CPU、小模型、可降採樣；Phase 5 前不優化 |
| 3 | 跳躍造成的 ~1 m 位置誤差 | airborne 偵測與旗標；不以平滑掩蓋 |
| 4 | 禁區擁擠造成 ID 互換 | 球場座標關聯 + 外觀；以 IDSW 量測 |
| 5 | 裁判、板凳、觀眾誤判為球員 | 場內遮罩 + 裁判類別；以 precision 量測 |
| 6 | 特殊地板設計（City Edition、彩繪地板）使線條偵測失效 | 地標模型為主、線條為輔；分球場回報 |
| 7 | 記分板、廣告圖層遮擋 | 固定圖層遮罩 |
| 8 | 轉播鏡頭畸變、非 PTZ 機位 | k1 參數；C 一致性檢查判定機位 |
| 9 | 球體小、動態模糊 | Phase 3 專用偵測器；MVP 誠實報告 |
| 10 | 影片與資料集授權 | 見 §10 |

### 9.2 驗證資料與標註流程
1. 選 20–30 秒主機位片段作為 **held-out 驗證集**（不用於調參）。
2. **CVAT** 標註球員 bbox + 身分（關鍵幀每 5–10 幀，中間內插後人工修正），匯出 MOT 格式 → 偵測 P/R、HOTA、IDF1、IDSW。
3. **地標**：每 30 幀用本專案工具點選所有可見地標 → 一半用於擬合、一半 held-out 量重投影誤差（避免自我驗證）。
4. **腳部地面點**：同一批幀上點選每位球員的腳底接觸點 → 以「人工校正 + 人工腳點」為真值，量公尺誤差。
5. **標註雜訊**：10% 幀重複標註，估計真值本身的誤差下限。
6. **合成資料**：已知相機參數的渲染影像 → 幾何模組必須精確還原（單元測試）。
7. **物理合理性**：速度 > 10 m/s、加速度異常、出界跳變的比例。
8. **籃框重投影**：3D 籃框投影與偵測位置的殘差（獨立校正檢驗）。
9. 偵測、追蹤、座標重建三者**分開報告**；座標誤差另以「真值框 + 系統校正」與「系統框 + 系統校正」兩種設定拆解誤差來源。
10. 選用外部真值：SportsMOT 含 NBA 轉播片段與追蹤真值（授權需確認，預期為非商業研究用）可做追蹤的外部基準；公開的 2015-16 SportVU 資料理論上可提供公尺級真值，但與轉播影片的對齊與授權不確定，列為後續選項。

---

## 10. 授權與資料來源

- NBA 轉播影片受著作權保護。個人研究用的短片段通常較能被接受，但**影片需由你提供並自行確認使用合法性**；本專案不會下載或散布 NBA 影片。
- 模型：優先 Apache 2.0 / MIT（RF-DETR N–L、supervision、rtmlib 待確認）。AGPL（Ultralytics、boxmot）僅在你同意後作為比較基線。
- 資料集：Roboflow 球場地標資料集授權需在下載頁確認；DeepSportRadar 為 CC BY-NC 系列（僅研究用）。

---

## 11. 需要你決定的事項

1. **測試影片**：請提供一段 30–60 秒、以主機位為主、720p/1080p 的 NBA 轉播 MP4（放在 `data/raw/`）。
2. **訓練算力**：是否同意在 Google Colab（免費版或 Pro 約 US$10/月）上微調球場地標模型？或先只做半自動校正？
3. **授權立場**：是否允許使用 AGPL 套件（Ultralytics YOLO）作為比較基線？
4. **環境位置**：建立 conda env 於 D 槽（建議），約需 3–5 GB。

---

## 參考來源
- KaliCalib: https://arxiv.org/pdf/2209.07795
- PnLCalib: https://arxiv.org/abs/2404.08401
- TVCalib: https://arxiv.org/pdf/2207.11709
- SportsMOT: https://arxiv.org/pdf/2304.05170
- Deep-EIoU: https://arxiv.org/pdf/2306.13074
- Deep HM-SORT（追蹤器比較表）: https://arxiv.org/pdf/2406.12081
- Basketball-SORT: https://arxiv.org/pdf/2406.19655
- Roboflow RF-DETR keypoint 教學: https://roboflow.com/blog/train-rf-detr-keypoint
- Roboflow 籃球追蹤教學: https://blog.roboflow.com/identify-basketball-players/
- Roboflow 三秒違例（YOLO 地標 + homography）: https://blog.roboflow.com/detect-3-second-violation-ai-basketball/
- RF-DETR 授權: https://blog.roboflow.com/rf-detr-is-free-to-use-commercially/
- roboflow/sports: https://github.com/roboflow/sports
- rtmlib: https://pypi.org/project/rtmlib/
- DeepSportRadar: https://arxiv.org/abs/2208.08190 、 https://github.com/gabriel-vanzandycke/deepsport
- AMD ROCm Windows 相容性: https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.1/docs/compatibility/compatibilityryz/windows/windows_compatibility.html
