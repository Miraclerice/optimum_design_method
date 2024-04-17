# -*- coding: utf-8 -*-
# @Author: MiracleRice
# Blog   : miraclerice.com
import os
import sys

import numpy as np
import json

import requests


def load_city_coord(path):
    """加载城市坐标 path：txt_path"""
    city_coord = []
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('  ')
            city_coord.append([float(line[0]), float(line[1])])
    return np.asarray(city_coord, dtype=np.float32)


def load_China_coord(path):
    """
    加载中国各省会、自治区、直辖市、特别行政区城市坐标
     path：json_path
     json结构为{'name': [{city_key: city_name, coord_key: [x, y]}...]}
     """
    city_coord = []
    with open(path, 'r', encoding='utf-8') as f:
        f = json.load(f)
        coord_name = list(f.keys())
        assert len(coord_name) == 1, f'{path} structure must be name: [city_key: city_name, coord_key: [x, y]...]'
        coord_lst = f[coord_name[0]]
        assert len(coord_lst) != 0, f'{path} must have at least one coord'
        for i in range(len(coord_lst)):
            keys = list(coord_lst[i].keys())
            coord_key = keys[1]
            coord = coord_lst[i][coord_key]
            city_coord.append([float(coord[0]), float(coord[1])])
    return np.asarray(city_coord, dtype=np.float32)


def load_random_coord(city_nums):
    """随机生成城市坐标"""
    city_coord = []
    for i in range(city_nums):
        city_coord.append([np.random.randint(0, 200), np.random.randint(0, 200)])
    return np.asarray(city_coord, dtype=np.float32)


class Logger(object):
    def __init__(self, save_dir, filename="log.txt"):
        self.terminal = sys.stdout
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.log = open(os.path.join(save_dir, filename), "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
        # self.log.flush()


def get_coord_by_geo(location):
    """
    获取指定位置坐标
    api: https://lbs.amap.com/api/webservice/guide/api/georegeo
    :return
    (标准城市名称, [经度, 纬度])
    """
    url = 'https://restapi.amap.com/v3/geocode/geo'
    params = {
        'key': 'your api key',
        'address': location
    }
    response = requests.get(url, params=params)
    data = response.json()
    if data['status'] == '1' and data['count'] != 0:
        return data['geocodes'][0]['formatted_address'], data['geocodes'][0]['location'].split(',')
    else:
        return None


"""
中国的省级行政区行政中心 
https://baike.baidu.com/item/%E4%B8%AD%E5%8D%8E%E4%BA%BA%E6%B0%91%E5%85%B1%E5%92%8C%E5%9B%BD%E7%9C%81%E7%BA%A7%E8%A1%8C%E6%94%BF%E5%8C%BA/54127472
"""
admin_center = ['北京市', '天津市', '	石家庄市', '太原市', '呼和浩特市', '沈阳市', '长春市',
                '哈尔滨市', '上海市', '南京市', '杭州市', '合肥市', '福州市', '南昌市', '济南市',
                '郑州市', '武汉市', '长沙市', '广州市', '南宁市', '海口市', '重庆市', '成都市', '贵阳市',
                '昆明市', '拉萨市', '西安市', '兰州市', '西宁市', '银川市', '乌鲁木齐市', '香港', '澳门',
                '台湾省台北市']


def save_admin_center_coord(city_lst):
    """保存中国的省级行政区行政中心"""
    ac_coord = {'China_admin_center_coord': []}
    for city in city_lst:
        coord = get_coord_by_geo(city)
        ac_coord['China_admin_center_coord'].append({'city': coord[0], 'coord': coord[1]})
        with open('China_admin_center_coord.json', 'w', encoding='utf-8') as f:
            json.dump(ac_coord, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    # save_admin_center_coord(admin_center)
    arr1 = load_China_coord('China_coord.json')
    print(arr1.shape)
    arr2 = load_China_coord('China_admin_center_coord.json')
    print(arr2.shape)
