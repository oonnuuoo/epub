import os
import sys
import zipfile
import shutil
from PIL import Image

def rename_files_in_directory_odd(directory, odd_n):
    # get jpeg files in directory
    files = [f for f in os.listdir(directory) if f.endswith('.jpg')]

    # sort files by name
    files.sort()

    # rename files until n-1 th file
    n = int(odd_n)
    for i, file in enumerate(files[:n-1], start=1):
        new_name = f"000_{i * 2 - 1:03}.jpg"
        os.rename(os.path.join(directory, file), os.path.join(directory, new_name))

    # rename files from n th file incrementally
    n = int(odd_n)-1
    for i, file in enumerate(files[n:], start=1):
        new_name = f"{i * 2 - 1:03}.jpg"
        os.rename(os.path.join(directory, file), os.path.join(directory, new_name))

def rename_files_in_directory_even(directory, even_n):
    # get jpeg files in directory
    files = [f for f in os.listdir(directory) if f.endswith('.jpg')]

    # sort files by name in descending order
    files.sort(reverse=True)

    # rename files until n-1 th file
    n = int(even_n)
    for i, file in enumerate(files[:n-1], start=1):
        new_name = f"000_{i * 2:03}.jpg"
        os.rename(os.path.join(directory, file), os.path.join(directory, new_name))

    # rename files from n th file decrementally
    n = int(even_n)-1
    for i, file in enumerate(files[n:], start=1):
        new_name = f"{i * 2:03}.jpg"
        os.rename(os.path.join(directory, file), os.path.join(directory, new_name))

def rotate_images_in_directory(directory):
    # get jpeg files in directory
    files = [f for f in os.listdir(directory) if f.endswith('.jpg')]

    for file in files:
        file_path = os.path.join(directory, file)
        with Image.open(file_path) as img:
            # rotate image 180 degrees
            rotate_img = img.rotate(180)
            # save rotated image
            rotate_img.save(file_path)


def unzip_zip_file(zip_file_path, folder_name=None):
    """ZIP を指定フォルダに展開し、odd/even フォルダを作成してフォルダ名を返す"""
    # フォルダ名が指定されていなければ、ZIP ファイル名から決定
    if not folder_name:
        folder_name = zip_file_path.replace('.zip', '')

    # unzip zip file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(folder_name)

    # make odd and even directories
    os.makedirs(folder_name + '/odd', exist_ok=True)
    os.makedirs(folder_name + '/even', exist_ok=True)

    return folder_name


def main():
    """main function"""
    print("=" * 60)

    # input zip file path to process
    zip_file_path = input("\n input zip file path to process: ").strip()

    # remove quote characters
    zip_file_path = zip_file_path.strip('"\'')

    # 展開先フォルダ名をユーザーに確認（未入力なら ZIP 名と同じにする）
    default_folder_name = zip_file_path.replace('.zip', '')
    folder_input = input(f"\n 画像を格納するフォルダ名を入力（空 Enter で '{default_folder_name}' ）: ").strip()
    if folder_input == "":
        folder_name = default_folder_name
    else:
        folder_name = folder_input

    folder_name = unzip_zip_file(zip_file_path, folder_name)
    print("\n unzipped folder name: " + folder_name)


    # divide images into odd and even
    number_of_images_odd = int(input("\n 奇数ページの枚数を入力 :  "))
    odd_n = int(input("\n 奇数ページの開始番号を入力 :  "))
    number_of_images_even = int(input("\n 偶数ページの枚数を入力 :  "))
    even_n = int(input("\n 偶数ページの開始番号を入力 :  "))
    odd_first = input("\n 奇数ページが最初に来るかどうかを入力 (y/n) :  ")
    rotate_folder = input("\n 回転するフォルダを入力 (odd/even) :  ")

    # move images to odd and even directories
    if odd_first == 'y':
        # move odd images to odd directory
        # sort files by name in ascending order and move number_of_images_odd files to odd directory
        files = [f for f in os.listdir(folder_name) if f.endswith('.jpg')]
        files.sort()

        print("PARAM folder_name: " + folder_name)
        
        for i in range(number_of_images_odd):
            shutil.move(folder_name + '/' + files[i], folder_name + '/odd/' + files[i])
            print("\n moved " + files[i] + " to odd directory")
            print(i)
            # print(files[i])
        # rename files in odd directory
        rename_files_in_directory_odd(folder_name + '/odd', odd_n)

        # move even images to even directory
        # sort files by name in descending order and move number_of_images_even files to even directory
        files = [f for f in os.listdir(folder_name) if f.endswith('.jpg')]
        files.sort(reverse=True)
        for i in range(number_of_images_even):
            shutil.move(folder_name + '/' + files[i-1], folder_name + '/even/' + files[i-1])

        rename_files_in_directory_odd(folder_name + '/odd', odd_n)
        rename_files_in_directory_even(folder_name + '/even', even_n)
    
    else:
        # move even images to even directory
        # sort files by name in ascending order and move number_of_images_even files to even directory
        files = [f for f in os.listdir(folder_name) if f.endswith('.jpg')]
        files.sort()
        for i in range(number_of_images_even):
            shutil.move(folder_name + '/' + files[i-1], folder_name + '/even/' + files[i-1])
        
        # move odd images to odd directory
        # sort files by name in ascending order and move number_of_images_odd files to odd directory
        files = [f for f in os.listdir(folder_name) if f.endswith('.jpg')]
        files.sort(reverse=True)
        for i in range(number_of_images_odd):
            shutil.move(folder_name + '/' + files[i-1], folder_name + '/odd/' + files[i-1])

        # rename files in odd directory
        rename_files_in_directory_odd(folder_name + '/odd', odd_n)
        rename_files_in_directory_even(folder_name + '/even', even_n)

    rotate_images_in_directory(folder_name + '/' + rotate_folder)
    print("\n 回転が完了しました")

if __name__ == "__main__":
    main()