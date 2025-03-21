import os
import sys
# add project directory to PATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from utils_general import load_dictionary_from_hdf5, save_dictionary_to_hdf5

import numpy as np

from tqdm import tqdm
from utils.macros import *
from utils.utils import *
from utils_general import get_directory_contents
import argparse
import warnings 
import os


# def find_character_following_string(text, target_string):
#     # Find the index of the target string in the text
#     index = text.find(target_string)

#     # Check if the target string is found in the text
#     if index != -1 and index + len(target_string) < len(text):
#         # If found, return the character following the target string
#         return text[index + len(target_string)]
#     else:
#         # If the target string is not found or there's no character following it, return None
#         return None
    

def infer_object_cls_from_path(path):
    # Find the index of 'box' in the path
    box_idx = path.find('box')

    if box_idx != -1:
        # Find the character following 'box'
        box_id_char = path[box_idx + 3]

        # Check if the character is a digit
        if box_id_char.isdigit():
            # Convert it to an integer and return
            return -1
            # return int(box_id_char)

    # Return -1 when 'box' is not found or the character cannot be converted
    return -1


total_num_compensation = 0
def annotate_dics_folder(src_dir, expected_num_object):
    """
    Annotate the dictionary folder with object class
    """
    dic_paths = get_directory_contents(src_dir)
    for dic_path in tqdm(dic_paths):
        if '.h5' not in dic_path:
             continue
        if 'processed' in dic_path:
            continue
        
        dic = load_dictionary_from_hdf5(dic_path)
        obj_cls = infer_object_cls_from_path(dic_path)
        new_dic = dic.copy()
        new_dic['object_cls'] = np.zeros(len(dic['object_pcs'])) + obj_cls
        
        # check num of objects
        num_steps, actual_num_object, N, pos_len = new_dic['object_pcs'].shape
        if expected_num_object is not None:
            if actual_num_object < expected_num_object:
                new_object_pcs = np.concatenate([new_dic['object_pcs'].copy(), np.zeros((num_steps, expected_num_object - actual_num_object, N, pos_len))], axis=1)
                new_dic['object_pcs'] = new_object_pcs
                print(f"actual_num_object {actual_num_object} < expected_num_object {expected_num_object}, compensated. ")
                global total_num_compensation
                total_num_compensation += 1
            elif actual_num_object > expected_num_object:
                warnings.warn(f'actual num of objects {actual_num_object} larger than expected num of objects {expected_num_object}')
            
        new_dict_path = dic_path.replace(args.data_prefix, f'{args.data_prefix}_anno')
        os.makedirs(os.path.dirname(new_dict_path), exist_ok=True)
        save_dictionary_to_hdf5(new_dic, new_dict_path)
        print(f'saved to {new_dict_path}')


def main(args):
    data_dir_prefix = args.data_prefix
    src_dir = os.path.join(DATA_DIR, data_dir_prefix)
    assert os.path.exists(src_dir), f'{src_dir} does not exist'

    annotate_dics_folder(src_dir, args.num_objects)

    print(f'Completed. Total number of compensated files: {total_num_compensation}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Your script description here.")
    parser.add_argument('--data-prefix', type=str, help='Prefix for data')
    parser.add_argument('--num-objects', type=int, help='Number of objects')
    args = parser.parse_args()

    main(args)
