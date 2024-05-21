import os
import pandas as pd
# import chardet
import csv
import re
from typing import List
from PIL import Image


def remove_blank_page(cur_path,pictures_list:List[str]):
    # print(f"The initial list of pictures: {pictures_list}")
    for picture in pictures_list:
        # print("picture",picture)
        if(picture.find('0001')!=-1):
            pictures_list.remove(picture)
            delete_file(cur_path+picture)
            # print(f"{picture} is removed.")

def delete_file(file_path):
    if os.path.isfile(file_path):
        os.remove(file_path)
    else:
        print(f"Error: {file_path} not a valid filename")

def rename_picture(folder_path,new_folder_path,new_name: str):
    global file_index
    if file_index > len(image_files_list):
        exit()
    # print("image_files_list",image_files_list)
    print("file_index",file_index)
    old_name = image_files_list[file_index]
    print(f"{file_index},file_index;\n{new_name},new name")
    print("old_name",old_name)
    pic_org = Image.open(os.path.join(folder_path, old_name)) 
    picture_path = os.path.join(new_folder_path, new_name)
    print("picture_path",picture_path)
    pic_org.save(picture_path)
    file_index += 1

def rename_all_pictures_book3(folder_path,new_folder_path,df):
    reference_list: List = list(zip(df['Image_ID'], df['cn_caption'], df['en_caption'], df['is_multipic']))
    result = []
    for item in reference_list:
        print("\nitem",item)
        if (item[3] == 'Y'):
            pattern_is_multi2one = r'[A-Z][\.．]'
            # match = re.search(pattern_is_multi2one, str(item[1]))
            match = re.findall(pattern_is_multi2one, str(item[1]))
            if (match):
                print("match",match)
                letter_pre = ''
                for c in match:
                    letter_cur = c.split('．')[0]
                    print("letter_cur",letter_cur)
                    letter_cur = letter_cur.split('.')[0]
                    print("letter_cur",letter_cur)
                    if letter_pre == '':
                        rename_picture(folder_path,new_folder_path,item[0]+letter_cur+'.jpg')
                        temp = (item[0]+letter_cur+'.jpg', item[1], item[2], item[3])
                        result.append(temp)
                    else:
                        if ord(letter_cur) - ord(letter_pre) == 1:
                            rename_picture(folder_path,new_folder_path,item[0]+letter_cur+'.jpg')
                            temp = (item[0]+letter_cur+'.jpg', item[1], item[2], item[3])
                            result.append(temp)
                        else:
                            continue
                    letter_pre = letter_cur
        else:
            rename_picture(folder_path,new_folder_path,item[0]+'.jpg')
            temp = (item[0]+'.jpg', item[1], item[2], item[3])
            result.append(temp)

    # print("result",result)
    print("len(result)",len(result))
    return result

def rename_all_pictures_book4(folder_path,new_folder_path,df):
    reference_list: List = list(zip(df['Image_ID'], df['cn_caption'], df['en_caption'], df['is_multipic']))
    result = []
    for item in reference_list:
        print("\nitem",item)
        if (item[3] == 'Y'):
            pattern_is_multi2one = r'[A-Z]\.'
            # match = re.search(pattern_is_multi2one, str(item[1]))
            match = re.findall(pattern_is_multi2one, str(item[2]))
            if (match):
                print("match",match)
                letter_pre = ''
                for c in match:
                    letter_cur = c.split('.')[0]
                    # print("letter_cur",letter_cur)
                    letter_cur = letter_cur.split('.')[0]
                    # print("letter_cur",letter_cur)
                    if letter_pre == '':
                        rename_picture(folder_path,new_folder_path,item[0]+letter_cur+'.jpg')
                        temp = (item[0]+letter_cur+'.jpg', item[1], item[2], item[3])
                        result.append(temp)
                    else:
                        if ord(letter_cur) - ord(letter_pre) == 1:
                            rename_picture(folder_path,new_folder_path,item[0]+letter_cur+'.jpg')
                            temp = (item[0]+letter_cur+'.jpg', item[1], item[2], item[3])
                            result.append(temp)
                        else:
                            continue
                    letter_pre = letter_cur
        else:
            rename_picture(folder_path,new_folder_path,item[0]+'.jpg')
            temp = (item[0]+'.jpg', item[1], item[2], item[3])
            result.append(temp)

    # print("result",result)
    print("len(result)",len(result))
    return result


if __name__ == "__main__":
    book_path = './book3/'  #replace
    total_part_number = 3   #replace (the number of sub-PDFs)

    for i in range(total_part_number):
        csv_path = book_path + 'caption/part' + str(i+1) + '.csv'
        folder_path = book_path + 'images_raw/'
        new_folder_path = book_path + 'images/'
        output_path = book_path + 'caption/part' + str(i+1) + '_final.csv'

        df = pd.read_csv(csv_path,encoding='utf-8')
            # './part1/caption/part3.csv',encoding='gb2312')

        all_files_list = os.listdir(folder_path)
        image_files_list = [file for file in all_files_list if file.endswith(
            ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]
        file_index = 0
        image_files_list = sorted(image_files_list)

        result = []
        remove_blank_page(folder_path,image_files_list)
        result = rename_all_pictures_book3(folder_path,new_folder_path,df)      #replace

        with open(output_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Image_ID", "cn_caption", "en_caption", "is_multipic"])

            for item in result:
                writer.writerow([item[0], item[1], item[2], item[3]])
