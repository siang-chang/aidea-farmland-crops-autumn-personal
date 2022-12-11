# Data

## train_split_fullset.pkl
- [train_split_fullset.pkl](train_split_fullset.pkl) 為 [train_tag_loc_coor_describe_elevation.csv](train_tag_loc_coor_describe_elevation.csv) 完整版，資料筆數 **89514**，並預先以 **0.7:0.15:0.15** 的比例切分為 training、validation 與 testing set
- 切分資料時皆有比照 `label`、`county_name` 欄位的分布
- 2022.10.22 新增影像角度資訊，並修正寬度與高度


## train_split_subset.pkl
- [train_split_subset.pkl](train_split_subset.pkl) 為 [train_tag_loc_coor_describe_elevation.csv](train_tag_loc_coor_describe_elevation.csv) 之子集合 (**20%**)，資料筆數 **17899**，並預先以 **0.7:0.15:0.15** 的比例切分為 training、validation 與 testing set
- 2022.10.22 新增影像角度資訊，並修正寬度與高度
- 切分資料時皆有比照 `label`、`county_name` 欄位的分布

    | 欄位名稱        | 欄位說明             |
    | --------------- | -------------------- |
    | file            | 檔案名稱             |
    | label           | 作物類別             |
    | target\_fid     | 流水號               |
    | target\_x       | 影像中心點 x 座標    |
    | target\_y       | 影像中心點 y 座標    |
    | county\_name    | 縣市名稱             |
    | town\_name      | 鄉鎮名稱             |
    | town\_x         | 鄉鎮 x 座標          |
    | town\_y         | 鄉鎮 y 座標          |
    | town\_z         | 鄉鎮海拔[^1]         |
    | make            | 拍攝相機的製造商[^2] |
    | model           | 拍攝相機的型號[^2]   |
    | taken\_datetime | 影像拍攝的時間[^2]   |
    | taken\_year     | 影像拍攝的年份[^2]   |
    | taken\_month    | 影像拍攝的月份[^2]   |
    | taken\_hour     | 影像拍攝的小時[^2]   |


[^1]: 使用 [Open-Elevation](https://open-elevation.com/) 服務計算得到  
[^2]: 影像描述資料，可能存在缺值  

## train_tag_loc_coor_describe_elevation.csv, public_tag_loc_coor_describe_elevation.csv
- [train_tag_loc_coor_describe_elevation.csv](train_tag_loc_coor_describe_elevation.csv) 在資料 [train_tag_loc_coor_describe.csv](train_tag_loc_coor_describe.csv) 再加入海拔訊息，每筆資料的海拔使用屬性 `town_x` 及 `town_y` 以 [Open-Elevation](https://open-elevation.com/) 服務計算
- 範例 https://api.open-elevation.com/api/v1/lookup?locations=25.071182,121.781205

    ```python
    # script for returning elevation from lat, long, based on open elevation data, which in turn is based on SRTM
    def get_elevation(lat, long):
        query = ('https://api.open-elevation.com/api/v1/lookup'
                f'?locations={lat},{long}')
        r = requests.get(query).json()  # json object, various ways you can extract value
        # one approach is to use pandas json functionality:
        elevation = pd.json_normalize(r, 'results')['elevation'].values[0]
        return elevation
    ```
- [public_tag_loc_coor_describe_elevation](public_tag_loc_coor_describe_elevation.csv) 公開階段使用
- 2022.10.22 新增影像角度資訊，並修正寬度與高度

## train_tag_loc_coor_describe.csv, public_tag_loc_coor_describe.csv
- [train_tag_loc_coor_describe.csv](train_tag_loc_coor_describe.csv) 在官方提供資料 [train_tag_loc_coor.csv](train_tag_loc_coor.csv) 再加入影像描述資料，需要注意該描述資料並非每張影像都有，可能存在缺值
- [public_tag_loc_coor_describe](public_tag_loc_coor_describe.csv) 公開階段使用
- 2022.10.22 新增影像角度資訊，並修正寬度與高度
- 舊版的寬度與高度有被 PIL 根據角度自動調換，現在調換成 CV2 讀出來的 shape
- 角度資訊從影像描述資料中取得，參考[資料](https://stackoverflow.com/questions/13872331/rotating-an-image-with-orientation-specified-in-exif-using-python-without-pil-in)
- 從描述資料取得的角度是設備在拍攝當下自動判斷的角度，所有仍有出錯的可能，但比例不高，以下是每種類別隨機抽取 10 張進行修正的結果[^3]
    |              | precision | recall | f1-score | support  |
    | ------------ | --------- | ------ | -------- | -------- |
    | 0            | 0.947     | 0.939  | 0.943    | 246      |
    | 90           | 0.814     | 0.897  | 0.854    | 78       |
    | 180          | 1.000     | 0.000  | 0.000    | 1        |
    | 270          | 1.000     | 0.000  | 0.000    | 5        |
    | accuracy     | 0.912     | 0.912  | 0.912    | 0.912121 |
    | macro avg    | 0.940     | 0.459  | 0.449    | 330      |
    | weighted avg | 0.916     | 0.912  | 0.905    | 330      |

[^3]: 每種類別隨機抽取 10 張，先由人工標註角度後，再與描述資料提供的角度進行比對  

## train_tag_loc_coor.csv, public_tag_loc_coor
- [train_tag_loc_coor.csv](train_tag_loc_coor.csv) 為官方提供的原始資料，訓練階段使用
- [public_tag_loc_coor.csv](train_tag_loc_coor.csv) 為官方提供的原始資料，公開驗證階段使用
