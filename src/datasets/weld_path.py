# Created by Xufeng Huang on 2022-05-24
# Email: xufenghuang1228@gmail.com

import os

from src.utils.string_operate import str_join_list

# conditions = ['N15_M01_F10', 'N15_M07_F04', 'N15_M07_F10', 'N09_M07_F10']  # B1, B2, B3, B4
_sensors = ['acoustic', 'photodiode']

_types_total = ['1-1', '1-2', '1-3', '1-4', '1-5',  # over
                '2-1', '2-2', '2-3', '2-4', '2-5',  # normal
                '3-1', '3-2', '3-3', '3-4', '3-5']  # under
#
_over = ['1-1', '1-2', '1-3', '1-4', '1-5']
_normal = ['2-1', '2-2', '2-3', '2-4', '2-5']
_under = ['3-1', '3-2', '3-3', '3-4', '3-5']

# _over = ['1-1']
# _normal = ['2-1']
# _under = ['3-1']

_acoustic_over = str_join_list(_sensors[0], _over)
_photodiode_over = str_join_list(_sensors[1], _over)
Over = (_acoustic_over, _photodiode_over)

_acoustic_normal = str_join_list(_sensors[0], _normal)
_photodiode_normal = str_join_list(_sensors[1], _normal)
Normal = (_acoustic_normal, _photodiode_normal)

_acoustic_under = str_join_list(_sensors[0], _under)
_photodiode_under = str_join_list(_sensors[1], _under)
Under = (_acoustic_under, _photodiode_under)

All = (Over, Normal, Under)

if __name__ == '__main__':
    print(len(All))
