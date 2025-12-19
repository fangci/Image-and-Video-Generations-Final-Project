import pandas as pd
import os
import glob

def merge_csv_files(folder_path, output_filename='merged_output.csv'):
    """
    合併指定資料夾內所有 CSV 檔案的內容。

    Args:
        folder_path (str): 包含 CSV 檔案的資料夾路徑。
        output_filename (str): 合併後輸出檔案的名稱。
    """
    # 1. 構建匹配所有 CSV 檔案的路徑
    # 使用 os.path.join 確保跨作業系統的相容性
    # 使用 glob.glob 來尋找所有匹配的文件
    search_pattern = os.path.join(folder_path, '*.csv')
    all_files = glob.glob(search_pattern)

    # 檢查是否有找到任何 CSV 檔案
    if not all_files:
        print(f"🚨 在資料夾 '{folder_path}' 中找不到任何 CSV 檔案。")
        return

    print(f"📦 找到 {len(all_files)} 個 CSV 檔案，準備合併...")

    # 2. 讀取並合併所有檔案
    # 創建一個空的列表來儲存每個 CSV 檔案的 DataFrame
    dataframes = []

    for filename in all_files:
        try:
            # 讀取 CSV 檔案
            # 這裡假設所有 CSV 檔案都使用相同的編碼 (utf-8)
            # 如果你有編碼問題，可能需要調整 'encoding' 參數
            df = pd.read_csv(filename)
            
            # (可選) 在 DataFrame 中新增一個欄位來標記資料來源
            # df['source_file'] = os.path.basename(filename) 
            
            dataframes.append(df)
            print(f"   ✅ 已讀取檔案: {os.path.basename(filename)} ({len(df)} 筆資料)")
            
        except Exception as e:
            print(f"   ❌ 讀取檔案 {os.path.basename(filename)} 時發生錯誤: {e}")


    # 3. 將所有 DataFrame 合併成一個
    # 使用 pd.concat 垂直堆疊所有 DataFrame
    if dataframes:
        merged_df = pd.concat(dataframes, ignore_index=True)
        
        # 4. 儲存合併後的 DataFrame 到新的 CSV 檔案
        output_path = os.path.join(folder_path, output_filename)
        merged_df.to_csv(output_path, index=False, encoding='utf-8')

        print("-" * 30)
        print(f"🎉 成功合併所有 CSV 檔案！")
        print(f"💾 總計資料筆數: {len(merged_df)}")
        print(f"📤 輸出檔案路徑: {output_path}")
    else:
        print("😥 沒有成功的 DataFrame 可以合併。")


# --- 使用範例 ---
if __name__ == '__main__':
    target_folder = '/home/moony/storage/fangci/AnimateDiff/webvid_csv' 
    
    # 輸出檔案名稱
    output_name = 'webvid.csv'

    # 執行合併功能
    merge_csv_files(target_folder, output_name)