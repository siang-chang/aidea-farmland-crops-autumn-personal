# Introduction

說明資料夾與程式檔案的用途

- `analysis` 存放資料分析的產出及程式檔案
- `data` 存放官方提供的資料及處理後的資料
- `image` 存放影像資料前處理的程式碼
- `logs` 存放模型訓練的紀錄檔案
- `models` 存放訓練完畢的模型及權重檔案
- `outputs` 存放各個階段的類別機率輸出檔案
- `sample` 存放少量影像樣本
- `submission` 存放 Public 及 Private 階段的提交檔案
- `utils` 存放公用程式
- `classification_binary_efficientnet.ipynb` 二元分類的深度學習模型程式
- `classification_multiclass_efficientnet.ipynb` 多類別的深度學習模型程式，這次競賽的主要模型
- `ensemble_xgb.ipynb` 集成模型的程式，這次競賽的次要模型
- `pre_processing.ipynb` 資料前處理程式，通常需要最先執行，不過目前會用到的資料都已經處理好了

# Method

說明主要的方法與架構，更多實驗細節紀錄在 `logs` 資料夾的文件

## Data Pre-Processing

說明資料前處理的方法，程式實作參考 [pre_processing.ipynb](./pre_processing.ipynb) 檔案

- 從官方提供的經緯座標計算海拔高度，更多細節參考[資料描述](./data#train_tag_loc_coor_describe_elevationcsv)
- 從影像的 EXIF 提取角度、拍攝時間及相機型號等屬性，更多細節參考[資料描述](./data#train_tag_loc_coor_describecsv)
- 將類別欄位轉換為 One-Hot Encoding 格式，供模型訓練使用
- 使用從 EXIF 中提取的角度資訊來修正影像角度，參考[角度修正實驗](#Image-Angle-Correction-by-Rule)
- 使用官方提供的準心資訊作為中心點來裁剪影像，裁剪比例為 70%，參考[影像裁切實驗](#Image-Crop-by-XY)
- 調整影像大小，縮放到相同的解析度（224x224）
- 模型架構帶有[正規化效果](https://www.tensorflow.org/api_docs/python/tf/keras/applications/efficientnet)，直接輸入值域為 0 到 255 之間的向量

## Data Split

說明不同階段的資料拆分策略，對於資料的更多細部資訊可以參考[資料描述](./data#data-description)章節

- 在 Training 階段使用全部資料集的 20% 進行小規模快速實驗，資料總共 17899 筆，預先以 70%、15%、15% 的比例切分為訓練（Training）、驗證（Validation）與測試（Testing）資料集
- 在 Public 階段使用全部資料進行完整實驗，資料總共 89514 筆，預先以 85%、15% 的比例切分為訓練（Training）與驗證（Validation）資料集，並以 Public Submission 代替測試（Testing）資料集
- 在 Public 階段的 Ensemble 實驗，將各個 Deep Learning 模型在驗證集（Validation）的輸出用於訓練 Ensemble 模型，並從其中拆分 20% 用於超參數優化

## Deep Learning Method: EfficientNet

模型使用 EfficientNet 在 Keras 上基於 ImageNet 之預訓練模型作為特徵擷取器，模型的 Dropout 比例為 40%，預訓練模型的權重僅作為模型初始權重，訓練過程中允許調整，並依序接上 1 層 GlobalAveragePooling2D 及 1 層有 33 個節點的全連接層作為分類層，模型架構如下所示：

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
efficientnetb3 (Functional)  (None, 7, 7, 1536)        10783535
_________________________________________________________________
avg_pool (GlobalAveragePooli (None, 1536)              0
_________________________________________________________________
pred (Dense)                 (None, 33)                50721
=================================================================
Total params: 10,834,256
Trainable params: 10,746,953
Non-trainable params: 87,303
_________________________________________________________________
```

訓練過程中，採用 Categorical Cross Entropy 作為 Loss Function；此外，若 Loss 在持續 10 個 Epoch 內沒有下降，就將 Learning Rate 設為當前的 0.31 倍，若 Loss 在持續 50 個 Epoch 內沒有下降，就停止訓練；訓練結束後，會將訓練階段在驗證集擁有最佳表現之 Epoch 的權重作為模型的最終權重，模型的其餘參數包括：

- BatchSize 設為 64
- 優化器使用 Adam
- 學習率設為 5e-4
- Epoch 設為 100
- Earlystop 機制設為 50 個 Epoch

## Ensemble Method: XGBoost

在這個比賽，我們分別使用了 4 種 Deep Learning 模型進行類別預測，我使用 [EfficientNet](https://www.tensorflow.org/api_docs/python/tf/keras/applications/efficientnet)、[@Tianming8585](https://github.com/Tianming8585) 使用 [Big Transfer](https://github.com/google-research/big_transfer)、[@Tsao666](https://github.com/Tsao666) 使用 [DCNN](https://arxiv.org/abs/2011.12960) 及 [ConvNeXt](https://www.tensorflow.org/api_docs/python/tf/keras/applications/convnext)，我們的模型各自有擅長與不擅長預測的類別，故使用 XGBoost（eXtreme Gradient Boosting）尋找模型間的最佳集成權重，集成模型的輸入參數是其他 Deep Learning 模型對 33 個類別的機率輸出，假設有 4 個模型參與集成，輸入參數就有 132 個向量

在訓練階段，使用其他 Deep Learning 模型對驗證資料集的輸出作為 XGBoost 的訓練資料，再從其中拆分 20% 用於超參數優化，筆數分別是 10727 筆與 2663 筆，在提交階段，則使用其他 Deep Learning 模型對 Public 或 Private 資料集的輸出作為輸入參數

下表為我們所有模型在 Public 資料集的績效比較表，可以發現 Ensemble Xgb 更進一步提升 Weighted Precision 分數

| Model         | Weighted Precision |
| :------------ | :----------------: |
| Ensemble Xgb  |       0.9226       |
| Big Transfer  |       0.9148       |
| ConvNeXt-Base |       0.8792       |
| EfficientNet  |       0.7814       |
| DCNN          |       0.7813       |

# Conclusion

在這個比賽，最終取得了 Public 前 11% (18/153) 及 Private 前 15% (23/153) 的成績

從 EfficientNet 在訓練與驗證階段的績效可以發現模型有 Overfitting 問題，過程中嘗試的幾種方法最終使測試集的 Weighted Precision 從 0.6637 提升至 0.7114，但距離完全解決 Overfitting 仍有不小距離，未來可以嘗試的改善方向有以下幾點：

- 嘗試使用其他資料增強方法，以及不同方法的搭配組合
- 嘗試使用正則化等模型訓練手段，降低擬合程度
- 嘗試使用較大的學習率，避免模型停在局部最佳解

使用 XGBoost 集成其他模型的預測結果可以顯著提升績效，在這個部分的未來改善方向則有以下幾點：

- 集成的時候不僅就現有模型進行權重調整，亦可以考慮移除部分績效較差的模型，這次在這個部分的實驗著墨較少
- 嘗試使用更大量的訓練資料，例如：總資料的 30%
