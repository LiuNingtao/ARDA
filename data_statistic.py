'''
@Author: your name
@Date: 2020-06-12 09:45:05
@LastEditTime: 2020-07-11 15:31:01
@LastEditors: ningtao liu
@Description: In User Settings Edit
@FilePath: /ToothAge/data_statistic.py
'''
import os
import pydicom
import datetime
from utils import get_gap
import pandas as pd
from collections import Counter
import random
import shutil


ROOT_DIR_list = []


def get_file_info():
    for root_dir in ROOT_DIR_list:
        data_dict = {
        'FILE_PATH': [],
        'ID': [],
        'BIRTH_DATE': [],
        'CONTENT_DATE': [],
        'AGE_DAY': [],
        'AGE_MONTH': [],
        'AGE_YEAR':[],
        'GENDER':[],
        'SHAPE': [],
        'TYPE':[],
        'NEW_NAME':[]
        }
        year = os.path.split(root_dir)[-1]

        path_list = os.walk(root_dir)
        for path, _, file_list in path_list:
            for file_name in file_list:
                file_path = os.path.join(path, file_name)
                assert os.path.exists(file_path)
                dicom_data = pydicom.dcmread(file_path)
                id_number = dicom_data.PatientID
                # 脱敏 不读取名字
                # name = dicom_data.PatientName
                birth_date = dicom_data.PatientBirthDate
                birth_date = birth_date.replace('.', '').replace('-', '').replace(' ', '')
                content_date = dicom_data.ContentDate
                content_date = content_date.replace('.', '').replace('-', '').replace(' ', '')
                gender = dicom_data.PatientSex
                age_days, age_month = get_gap(birth_date, content_date, r'%Y%m%d')
                age_year =  round(age_month / 12) if age_month != -1 else -1
                try:
                    shape = dicom_data.pixel_array.shape
                    if len(shape) > 2:
                        shape = shape[0: 2]
                    if shape[1] / shape[0] > 1:
                        data_type = 'JAW'
                    else:
                        data_type = 'SKULL'
                except:
                    shape = [-1, -1]
                    data_type = 'UNKNOW'
                
                data_dict['FILE_PATH'].append(file_path)
                data_dict['ID'].append(id_number)
                data_dict['BIRTH_DATE'].append(birth_date)
                data_dict['CONTENT_DATE'].append(content_date)
                data_dict['AGE_DAY'].append(age_days)
                data_dict['AGE_MONTH'].append(age_month)
                data_dict['AGE_YEAR'].append(age_year)
                data_dict['GENDER'].append('MALE' if gender != 'F' else 'FEMALE')
                data_dict['SHAPE'].append(str(shape[0]) + ',' + str(shape[1]))
                data_dict['TYPE'].append(data_type)

                id_number = id_number if id_number and len(id_number) > 0 else 'unknow'
                data_dict['NEW_NAME'].append(id_number + '_' + str(round(age_days / 365.25, 2)) + '_' + data_type +'.nii')

        data_df = pd.DataFrame(data_dict)
        data_df.to_csv('./{}_data_info.csv'.format(year), index=False)

def rename():
    """
    对数据进行重命名
    """
    csv_path_list = os.listdir('./')
    csv_path_list = [path for path in csv_path_list if 'data_info.csv' in path]
    for file_path in csv_path_list:
        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            origin_dir = row['FILE_PATH']
            contain_dir = os.path.split(origin_dir)[0]
            tar_dir = os.path.join(os.path.split(origin_dir)[0], row['NEW_NAME'])
            try:
                os.rename(tar_dir, origin_dir)
            except Exception as e:
                print(e)
                print(origin_dir)
                continue
        print(file_path)
    
    print('DONE')

def filter_patient_id():
    """
    筛选出有多张影像的病人
    """
    csv_path_list = os.listdir('./')
    csv_path_list = [path for path in csv_path_list if 'data_info.csv' in path]
    selected_dict = {
    
    'ID':[],
    'AGE_YEAR':[],
    'AGE_MONTH':[],
    'AGE_DAY':[],
    'TYPE':[],
    'FILE_PATH':[]}
    for csv_file in csv_path_list:
        df = pd.read_csv(csv_file)
        grouped = df.groupby(['AGE_YEAR','ID'])
        for index, _ in grouped:
            if int(index[0]) < 0:
                continue
            group_list = grouped.get_group((index[0], index[1]))
            if group_list.shape[0] != 2:
                continue
            type_list = group_list.TYPE.tolist()
            if 'JAW' in type_list and 'SKULL' in type_list:
                selected_dict['ID'].extend(group_list.ID.tolist())
                selected_dict['AGE_YEAR'].extend(group_list.AGE_YEAR.tolist())
                selected_dict['AGE_MONTH'].extend(group_list.AGE_MONTH.tolist())
                selected_dict['AGE_DAY'].extend(group_list.AGE_DAY.tolist())
                selected_dict['TYPE'].extend(group_list.TYPE.tolist())
                selected_dict['FILE_PATH'].extend(group_list.FILE_PATH.tolist())
    selected_df = pd.DataFrame(selected_dict)
    selected_df.to_csv('./selected_file.csv', index=False)

def get_choose_num():
    list_file_path = './selected_file.csv'
    file_df = pd.read_csv(list_file_path)
    grouped_age = file_df.groupby('AGE_YEAR')
    num_sum = file_df.shape[0]
    select_rate = 400 / num_sum

    age_num_dict = {'AGE_YEAR':[], 'NUM_SUM':[],'NUM_FLOAT':[], 'NUM_SELECT':[]}
    for index, _ in grouped_age:
        group_list = grouped_age.get_group(index)
        count = group_list.shape[0]
        age_num_dict['AGE_YEAR'].append(index)
        age_num_dict['NUM_SUM'].append(count)
        num_select = count * select_rate
        age_num_dict['NUM_FLOAT'].append(num_select)
        age_num_dict['NUM_SELECT'].append(round(num_select))
    age_num_df = pd.DataFrame(age_num_dict)
    age_num_df.to_csv('./num_for_select.csv', index=False)

def select_and_copy_data():
    copy_tar_dir = r''
    select_num_file = r''
    all_info_file = r''
    num_df = pd.read_csv(select_num_file)
    info_df = pd.read_csv(all_info_file)

    path_dict = {'AGE_YEAR':[], 'PATH':[]}
    grouped_age = info_df.groupby('AGE_YEAR')
    for _, row in num_df.iterrows():

        age = row['AGE_YEAR']
        num = row['NUM_SELECT']
        
        group = grouped_age.get_group(age)
        id_set = set(group.ID.tolist())
        id_selected = random.sample(id_set, int(num))
        selected_row = group[group.ID.isin(id_selected)]
        assert selected_row.shape[0] >= 2
        file_path_list = selected_row.FILE_PATH.tolist()
        path_dict['AGE_YEAR'].extend([age] * len(file_path_list))
        path_dict['PATH'].extend(file_path_list)
        for path in file_path_list:
            tar_path = os.path.join(copy_tar_dir, os.path.split(path)[-1])
            shutil.copy(path, tar_path)
            print(path)
    pd.DataFrame(path_dict).to_csv('./selected_path.csv')


    
if __name__ == "__main__":
    # get_file_info()
    # rename()
    # filter_patient_id()
    # get_choose_num()
    select_and_copy_data()