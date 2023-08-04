import shutil
import os
import glob
from sklearn.model_selection import train_test_split

# class ImageMove:
#     def __init__(self, org_folder):
#         self.org_folder = org_folder
#
#     def move_img(self):
#         file_path_list = glob.glob(os.path.join(self.org_folder,'*','*','*.png'))
#         for file_path in file_path_list:
#             folder_name = file_path.split('\\')[1]
#
#             if folder_name == 'MelSepctrogram':
#                 shutil.move(file_path, './data/ex_data/MelSepctrogram')
#             elif folder_name == 'STFT':
#                 shutil.move(file_path, './data/ex_data/STFT')
#             elif folder_name == 'waveshow':
#                 shutil.move(file_path, './data/ex_data/waveshow')
#             #print(folder_name)
#             #print(file_path)
#
#
# #test = ImageMove('./data/final_data')
# #test.move_img()

class ImageDataMove :

    def __init__(self, org_dir, train_dir, val_dir):

        self.org_dir = org_dir

        self.train_dir = train_dir

        self.val_dir = val_dir


    def move_images(self):
        #jpg!d out
    # file path list
        file_path_list01 = glob.glob(os.path.join(self.org_dir, "Abstract", "*.png"))
        file_path_list02 = glob.glob(os.path.join(self.org_dir, "Cubist", "*.png"))
        file_path_list03 = glob.glob(os.path.join(self.org_dir, "Expressionist", "*.png"))
        file_path_list04 = glob.glob(os.path.join(self.org_dir, "Impressionist", "*.png"))
        file_path_list05 = glob.glob(os.path.join(self.org_dir, "Landscape", "*.png"))
        file_path_list06 = glob.glob(os.path.join(self.org_dir, "Pop Art", "*.png"))
        file_path_list07 = glob.glob(os.path.join(self.org_dir, "Portrait", "*.png"))
        file_path_list08 = glob.glob(os.path.join(self.org_dir, "Realist", "*.png"))
        file_path_list09 = glob.glob(os.path.join(self.org_dir, "Still Life", "*.png"))
        file_path_list10 = glob.glob(os.path.join(self.org_dir, "Surrealist", "*.png"))
        print(file_path_list02)

    # data split
        ab_train_data_list , ab_val_data_list = train_test_split(file_path_list01, test_size=0.2)
        cu_train_data_list , cu_val_data_list = train_test_split(file_path_list02, test_size=0.2)
        ex_train_data_list , ex_val_data_list = train_test_split(file_path_list03, test_size=0.2)
        im_train_data_list, im_val_data_list = train_test_split(file_path_list04, test_size=0.2)
        la_train_data_list, la_val_data_list = train_test_split(file_path_list05, test_size=0.2)
        pop_train_data_list, pop_val_data_list = train_test_split(file_path_list06, test_size=0.2)
        po_train_data_list, po_val_data_list = train_test_split(file_path_list07, test_size=0.2)
        re_train_data_list, re_val_data_list = train_test_split(file_path_list08, test_size=0.2)
        st_train_data_list, st_val_data_list = train_test_split(file_path_list09, test_size=0.2)
        su_train_data_list, su_val_data_list = train_test_split(file_path_list10, test_size=0.2)

    # file move
        self.move_file(ab_train_data_list, os.path.join(self.train_dir, "Abstract"))
        self.move_file(ab_val_data_list, os.path.join(self.val_dir, "Abstract"))
        self.move_file(cu_train_data_list, os.path.join(self.train_dir, "Cubist"))
        self.move_file(cu_val_data_list, os.path.join(self.val_dir, "Cubist"))
        self.move_file(ex_train_data_list, os.path.join(self.train_dir, "Expressionist"))
        self.move_file(ex_val_data_list, os.path.join(self.val_dir, "Expressionist"))
        self.move_file(im_train_data_list, os.path.join(self.train_dir, "Impressionist"))
        self.move_file(im_val_data_list, os.path.join(self.val_dir, "Impressionist"))
        self.move_file(la_train_data_list, os.path.join(self.train_dir, "Landscape"))
        self.move_file(la_val_data_list, os.path.join(self.val_dir, "Landscape"))
        self.move_file(pop_train_data_list, os.path.join(self.train_dir, "Pop Art"))
        self.move_file(pop_val_data_list, os.path.join(self.val_dir, "Pop Art"))
        self.move_file(po_train_data_list, os.path.join(self.train_dir, "Portrait"))
        self.move_file(po_val_data_list, os.path.join(self.val_dir, "Portrait"))
        self.move_file(re_train_data_list, os.path.join(self.train_dir, "Realist"))
        self.move_file(re_val_data_list, os.path.join(self.val_dir, "Realist"))
        self.move_file(st_train_data_list, os.path.join(self.train_dir, "Still Life"))
        self.move_file(st_val_data_list, os.path.join(self.val_dir, "Still Life"))
        self.move_file(su_train_data_list, os.path.join(self.train_dir, "Surrealist"))
        self.move_file(su_val_data_list, os.path.join(self.val_dir, "Surrealist"))


    def move_file(self, file_list, mov_dir):

        os.makedirs(mov_dir, exist_ok=True)

        for file_path in file_list:

            shutil.move(file_path, mov_dir)

# org_dir = "./data/ex_data2_sorted"
# train_dir = "./data/art_data/train"
# val_dir = "./data/art_data/val"
# move_temp = ImageDataMove(org_dir, train_dir, val_dir)
# move_temp.move_images()


temp = 'Abstract'
org_dir = f'./data/ex_data2/{temp}'
file_path = glob.glob(os.path.join(org_dir,'*'))

import cv2
for path in file_path:
    file_type = path.split('.')[-1]
    file_name = path.split('\\')[1]
    file_name = file_name.split('.')[0]

    if file_type != 'jpg!d' and file_type != 'gif':
        os.makedirs(f'./data/ex_data2_sorted/{temp}', exist_ok=True)
        img = cv2.imread(path)
        cv2.imwrite(f'./data/ex_data2_sorted/{temp}/{file_name}.png', img)

