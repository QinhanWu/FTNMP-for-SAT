import numpy as np

def find_sublist_indices_with_nums(lst, num1, num2):
    indices = [i for i, sublist in enumerate(lst) if num1 in sublist and num2 in sublist]
    return indices

def find_matching_sublists(target_list, lst_2d):
    
    sorted_target = sorted(target_list)
    matching_indices = []


    for idx, sublist in enumerate(lst_2d):
     
        if sorted(sublist) == sorted_target:
            matching_indices.append(idx)
    
    return matching_indices

def merge_and_remove_duplicates(lst, indices):
   
    if len(indices) < 2:
        print("error!")
        return lst
    indices = sorted(indices) 
    merged_list = []

    for index in indices:
        merged_list.extend(lst[index])

    merged_list = list(set(merged_list))
    for index in indices:
        lst[index] = merged_list
    
    return lst

def merge_and_remove_2d_list(lst, indices):

    if len(indices) < 2:
        print("error")
        return lst
    
    indices = sorted(indices, reverse=True) 
    merged_list = []
    
    for index in indices:
        merged_list.extend(lst[index])
        del lst[index]
    merged_list = list(set(merged_list))
    
    insert_position = indices[-1]  
    lst.insert(insert_position, merged_list)
    
    return lst

def merge_and_remove_2d_list2(lst, indices):
    
    if len(indices) < 2:
        print("error!")
        return lst
    

    indices = sorted(indices, reverse=True) 
    merged_list = []
    
    for index in indices:
        merged_list.extend(lst[index])
        lst[index] = []
    merged_list = list(set(merged_list))
    

    insert_position = indices[-1]  
    lst[insert_position] = merged_list
    
    return lst


def find_all_indices_2d(lst, target):
    
    max_length = max(len(sublist) for sublist in lst)
    arr = np.array([sublist + [None] * (max_length - len(sublist)) for sublist in lst])
    
   
    indices = np.argwhere(arr == target)
    
    if len(indices) > 0:
        return indices.tolist()  
    else:
        return []
