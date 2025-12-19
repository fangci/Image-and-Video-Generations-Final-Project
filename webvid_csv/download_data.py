import pandas as pd
import os
import numpy as np
import argparse
import requests
import warnings 

# 移除對 mpi4py 的導入和初始化
# 移除 concurrent.futures 的導入

# 保持 request_save 函數不變，但增加錯誤處理
def request_save(url, save_fp):
    try:
        # 下載影片內容，設定超時 5 秒
        img_data = requests.get(url, timeout=5).content
        # 寫入檔案
        with open(save_fp, 'wb') as handler:
            handler.write(img_data)
        return True # 下載成功
    except requests.exceptions.RequestException as e:
        warnings.warn(f"下載失敗 (超時/連線錯誤): {url} to {save_fp}. 錯誤: {e}")
        return False # 下載失敗


def main(args):
    # 由於沒有 MPI，RANK 始終為 0，SIZE 始終為 1
    # 移除所有 COMM.barrier() 和 RANK/SIZE 相關檢查
    
    video_dir = os.path.join(args.data_dir, 'videos')
    
    # 只有一個程序，直接創建目錄
    if not os.path.exists(os.path.join(video_dir, 'videos')):
        os.makedirs(os.path.join(video_dir, 'videos'))
    
    print(f"✅ 正在讀取 CSV 檔案: {args.csv_path}")

    # 直接讀取完整的 CSV 文件，忽略分區邏輯
    try:
        df = pd.read_csv(args.csv_path)
    except FileNotFoundError:
        print(f"❌ 錯誤：找不到 CSV 檔案於 {args.csv_path}")
        return
    
    # --- 預處理邏輯 ---

    # 讀取已存在的影片清單（如果文件存在）
    relevant_fp = os.path.join(args.data_dir, 'relevant_videos_exists.txt')
    if os.path.isfile(relevant_fp):
        # 假設 relevant_videos_exists.txt 只有一列，無標頭
        try:
            exists_df = pd.read_csv(relevant_fp, names=['fn'], header=None)
            exists = set(exists_df['fn'].astype(str))
            print(f"ℹ️ 找到 {len(exists)} 個已存在的影片記錄，將跳過。")
        except pd.errors.EmptyDataError:
             exists = set()
    else:
        exists = set()

    # 創建相對路徑欄位
    df['rel_fn'] = df.apply(lambda x: os.path.join(str(x['page_dir']), str(x['videoid'])), axis=1)
    df['rel_fn'] = df['rel_fn'] + '.mp4'

    # 過濾已存在的影片
    df = df[~df['rel_fn'].isin(exists)]

    # 移除 page_dir 為空值 (NaN) 的行
    df.dropna(subset=['page_dir'], inplace=True)
    
    print(f"✅ 預計下載 {len(df)} 個影片。")

    # 按 page_dir 分組
    playlists_to_dl = np.sort(df['page_dir'].unique())
    total_downloaded = 0
    total_skipped = len(exists)
    
    # --- 依序下載循環 ---
    
    for page_dir in playlists_to_dl:
        pdf = df[df['page_dir'] == page_dir]
        
        if len(pdf) > 0:

            for idx, row in pdf.iterrows():
                video_fp = os.path.join(video_dir, str(row['videoid']) + '.mp4')
                
                # 再次檢查檔案是否存在，以防多程序環境中發生競爭（雖然我們是單程序）
                if os.path.isfile(video_fp):
                    total_skipped += 1
                    continue
                
                # 執行單線程依序下載
                is_success = request_save(row['contentUrl'], video_fp)
                
                if is_success:
                    total_downloaded += 1
                else:
                    # 失敗的記錄已在 request_save 中印出警告
                    pass

    print(f"\n--- 下載總結 ---")
    print(f"📥 成功下載數量: {total_downloaded}")
    print(f"⏭️  跳過數量 (已存在): {total_skipped}")
    print("下載完成。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Shutter Image/Video Downloader (Single Process)')
    # 僅保留下載必需的參數
    # 移除 --partitions 和 --part 參數
    parser.add_argument('--data_dir', type=str, default='/home/ado/storage/nas/webvid',
                        help='Directory where webvid data is stored.')
    parser.add_argument('--csv_path', type=str, default='webvid_csv/webvid.csv',
                        help='Path to csv data to download')
    # 移除 --processes 參數，因為不再使用線程池
    
    args = parser.parse_args()

    # 執行單程序主邏輯
    main(args)