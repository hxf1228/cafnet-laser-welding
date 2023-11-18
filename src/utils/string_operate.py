def str_join_list(x, y):
    file_list = []
    if not isinstance(x, list):
        for i in y:
            file_list.append(x + "_" + i)
    else:
        for j in x:
            file_list.append(j + "_" + y)
    return file_list
