
## 訓練所需之資料說明
- `TAD_202206.csv`
臺灣大氣事件資料庫 (https://osf.io/4zutj/)

- `FTdate_56_f1980.txt`
從TAD_202206.csv篩選出1980-2020年所有發生在5、6月之鋒面事件日期

- `1980050100.nc`
ERA5 Hourly Data是透過ECMWF官網提供之API下載。完整資料檔案過大，因此僅上傳其中一筆作為代表

- `vars_hr_f1980.npz`
此檔案將每日ERA5的資料合併為單一.npz檔案，以便後續讀取（檔案較大未上傳）

- `TCCIP_GriddedRain_TWN.nc`
TCCIP 網格化雨量資料（檔案較大未上傳）

- `TOPO.nc`
臺灣地形資料，僅用於雨量圖繪製（檔案較大未上傳）

## 訓練／視覺化過程產生之資料說明
- `Y_true_pred_f1980.txt`
此文件儲存正確類別／模型預測之類別

- `shap.npz`
將可解釋性詮釋結果輸出成單一.npz檔案，以便後續讀取（檔案較大未上傳）

- `shap_samples.txt`
此文件儲存欲進行詮釋結果之視覺化的樣本index


