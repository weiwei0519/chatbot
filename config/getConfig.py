# coding=UTF-8
# 项目工程中对配置文件进行读写维护的工具类
# 

'''
@File: getConfig.py
@Author: Wei Wei
@Time: 2022/7/30 10:26
@Email: weiwei_519@outlook.com
@Software: PyCharm
'''

from configparser import SafeConfigParser
from util.pathutil import PathUtil

project_path = PathUtil()


def get_config(config_file=project_path.rootPath + '/config/config.ini'):
    if config_file is None:
        config_file = project_path.rootPath + '/config/config.ini'
    cp = SafeConfigParser()
    cp.read(config_file)
    # 配置文件的格式是key-value形式
    _conf_ints = [(key, int(value)) for key, value in cp.items('ints')]
    # _conf_floats = [(key, float(value)) for key, value in cp.items('floats')]
    _conf_strings = [(key, str(value)) for key, value in cp.items('strings')]
    # 返回字典格式的配置文件
    return dict(_conf_ints + _conf_strings)
