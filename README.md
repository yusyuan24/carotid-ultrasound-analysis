# Carotid 影像處理流程

本資料夾包含頸動脈超音波 DICOM cine 的前處理、影像品質比較、血管腔分割與直徑／脈搏分析程式。大部分程式以 `__file__` 為基準尋找輸入與輸出資料夾，不需要另外指定命令列參數；執行前請確認程式所需的資料已放在對應位置。


## 環境需求

- Python 3.10 或更新版本
- Python 套件：`numpy`、`opencv-python`、`matplotlib`、`pydicom`、`Pillow`、`scipy`

可在虛擬環境中安裝依賴：

```bash
python -m pip install numpy opencv-python matplotlib pydicom Pillow scipy
```

## 資料與流程

預設 DICOM 輸入為 `CCA DICOM/00000000`。主要處理流程如下：

```text
CCA DICOM/00000000
  └─ crop_ultrasound.py
       └─ cropped_frames/frame_XXXX.png
            └─ alignment.py
                 └─ aligned_frames/frame_XXXX.png
                      └─ filter.py
                           └─ filter_results/<濾波方法>/frame_XXXX.png
                                └─ enhance_contrast.py
                                     └─ contrast_results/<濾波方法>/<增強方法>/frame_XXXX.png
                                          ├─ segmentation_rg_fcm_repair.py
                                          │    └─ segmentation_rg_fcm_repair_results/
                                          ├─ carotid_analysis.py
                                          │    └─ analysis_output/
                                          └─ segmentation_rg_fcm_repair_comparison.py
```

影格檔名採 `frame_0000.png` 格式。`crop_ultrasound.py` 會從 DICOM 讀取 cine、裁切固定範圍，並產生 `crop_metadata.json`，其中記錄影格數、裁切座標、像素間距與影格時間。後續分割與分析程式內的座標和比例係數是依目前 200×200 裁切影像設定；若更換資料或修改裁切區域，請一併檢查這些參數。

## 程式索引

| 程式 | 用途 | 主要輸出 |
|---|---|---|
| `crop_ultrasound.py` | 從 DICOM cine 擷取超音波畫面並依固定座標裁切。 | `cropped_frames/`、`crop_metadata.json` |
| `alignment.py` | 將裁切後影格以影像配準方式對齊，並記錄各影格位移。 | `aligned_frames/`、`alignment_shifts.csv` |
| `plot_alignment_check.py` | 讀取對位紀錄與 DICOM 資訊，繪製對位品質檢查圖。 | `analysis_output/alignment_check.png` |
| `plot_alignment_frame_comparison.py` | 將對位前後指定影格並列比較，並顯示位移資訊。 | `analysis_output/alignment_frame_comparison.png` |
| `filter.py` | 對所有對齊影格套用 Mean、Median、Lee、Butterworth 與 Gaussian 濾波，輸出影像及探索性指標。 | `filter_results/`，含 CSV 與文字摘要 |
| `filter_comparison.py` | 將 frame 0 的原圖與各濾波結果排版比較。 | `filter_results/filter_comparison_frame_0000.png` |
| `enhance_contrast.py` | 對各濾波影格套用多種對比增強方法，並彙整影像與指標。 | `contrast_results/`，含 CSV 與比較圖 |
| `enhance_contrast_comparison.py` | 比較 frame 0 的各濾波與對比增強結果。 | `contrast_results/` 下的比較圖 |
| `segmentation_comparison.py` | 在選定的前處理影格上比較多種血管腔分割方法。 | `segmentation_comparison_results/` |
| `segmentation_rg_fcm_hybrid.py` | 以 Region Growing 核心結合 FCM 邊緣資訊，分析指定代表影格。 | `segmentation_hybrid_results/frame_XXXX/` |
| `segmentation_rg_fcm_repair.py` | 結合 Region Growing 與 FCM，進行遮罩清理與缺口修補，並處理 cine 影格。 | `segmentation_rg_fcm_repair_results/`，含全影格遮罩、CSV、JSON 與檢視圖 |
| `segmentation_rg_fcm_repair_comparison.py` | 比較清理後遮罩與最終修補遮罩，並分析直徑、邊界梯度；可選擇讀取人工標記。 | `segmentation_rg_fcm_repair_comparison/` |
| `carotid_analysis.py` | 由分割遮罩估算逐影格管腔直徑、峰谷事件與脈搏壓力變化，並輸出品質檢查圖。 | `analysis_output/` |

## 建議執行順序

在專案根目錄開啟終端機，依序執行：

```bash
python crop_ultrasound.py
python alignment.py
python plot_alignment_check.py
python plot_alignment_frame_comparison.py
python filter.py
python filter_comparison.py
python enhance_contrast.py
python enhance_contrast_comparison.py
python segmentation_rg_fcm_repair.py
python carotid_analysis.py
python segmentation_rg_fcm_repair_comparison.py
```

`segmentation_comparison.py` 與 `segmentation_rg_fcm_hybrid.py` 是額外的分割實驗，可在對比增強完成後按需求執行。建議先查看各程式開頭的資料夾、代表影格、種子點、ROI、門檻與比例設定，再執行。這些腳本多數不是命令列工具，輸入路徑與分析參數需要在程式常數中調整。

## 輸出重點

- `aligned_frames/alignment_shifts.csv`：每一影格的對位位移與相關資訊。
- `filter_results/filter_metrics.csv`、`filter_metrics_summary.csv`：濾波結果的影像指標；這些是探索性品質指標，不是人工標註的準確度。
- `contrast_results/contrast_metrics.csv`、`contrast_metrics_summary.csv`：對比增強結果指標。
- `segmentation_rg_fcm_repair_results/all_frame_masks/`：供後續分析使用的逐影格分割遮罩。
- `segmentation_rg_fcm_repair_results/segmentation_metrics.csv`、`run_summary.json`：分割品質與執行摘要。
- `analysis_output/diameter_timeseries.csv`、`peak_valley_events.csv`、`estimated_pressure_timeseries.csv`：直徑與脈搏相關分析結果。

## 執行注意事項

1. 確認 DICOM 位於 `CCA DICOM/00000000`；若檔名或位置不同，請修改各腳本內的 `DICOM_PATH` 或 `SOURCE`。
2. 從前處理流程前段開始執行，確保下游所需的資料夾與影格已生成。多數腳本會建立自己的輸出資料夾，但不會自動補做缺少的前處理。
3. 比較與分析腳本採用程式內設定的輸入方法和影格索引；更換濾波方法、增強方法或裁切座標時，請同步調整相關設定。
4. `carotid_analysis.py` 使用 DICOM 的影格時間與程式設定的像素間距換算時間和毫米尺度，並依設定的收縮壓、舒張壓估算壓力波形；壓力值屬於此分析流程的估算結果。
5. 執行完成後可檢查各輸出資料夾中的 CSV、JSON、PNG，確認影格數與遮罩品質符合預期。
