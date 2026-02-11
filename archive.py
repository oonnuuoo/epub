import os
import shutil
import sys
import glob
from pathlib import Path

file_name = sys.argv[1]
file_name_1 = '../../' + file_name + '/odd'
file_name_2 = '../../' + file_name + '/even'
file_name_3 = '../../' + file_name + '/odd/cropped'
file_name_4 = '../../' + file_name + '/even/cropped'
file_name_5 = '../../' + file_name + '/margin_added'
file_name_6 = '../../' + file_name + '/comp'

def delete_jpg_files_basic(directory):

    deleted_count = 0

    if not os.path.exists(directory):
        print(f"⚠ ディレクトリ '{directory}' が見つかりません")
        return
    
    for filename in os.listdir(directory):
        if filename.lower().endswith('.jpg'):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                    deleted_count += 1
                except OSError as e:
                    print(f"Error deleting {file_path}: {e}")
    
    print(f"Total .jpg files deleted: {deleted_count}")



def move_to_archive():
    """
    Move A.zip file and A folder to archive directory
    """
    # Define source and destination paths
    source_file = Path(f"../../{file_name}.zip")
    source_folder = Path(f"../../{file_name}")
    archive_dir = Path("../../archive")
    
    try:
        # Create archive directory if it doesn't exist
        archive_dir.mkdir(exist_ok=True)
        
        # Move A.zip file if it exists
        if source_file.exists():
            destination_file = archive_dir / f"{file_name}.zip"
            shutil.move(str(source_file), str(destination_file))
            print(f"✓ ファイル '{source_file}' を '{destination_file}' に移動しました")
        else:
            print(f"⚠ ファイル '{source_file}' が見つかりません")
        
        # Move A folder if it exists
        if source_folder.exists() and source_folder.is_dir():
            destination_folder = archive_dir / file_name
            shutil.move(str(source_folder), str(destination_folder))
            print(f"✓ フォルダ '{source_folder}' を '{destination_folder}' に移動しました")
        else:
            print(f"⚠ フォルダ '{source_folder}' が見つかりません")
            
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")

if __name__ == "__main__":
    print("アーカイブ操作を開始します...")


    delete_jpg_files_basic(file_name_1)
    delete_jpg_files_basic(file_name_2)
    delete_jpg_files_basic(file_name_3)
    delete_jpg_files_basic(file_name_4)
    delete_jpg_files_basic(file_name_5)
    delete_jpg_files_basic(file_name_6)
    
    move_to_archive()
    print("アーカイブ操作が完了しました")
