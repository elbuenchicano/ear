import sys
import os
import argparse
import json
import re
import random 
import numpy as np
from sys import platform

################################################################################
################################################################################
def u_whichOS():
    if platform == "linux" or platform == "linux2":
        return 'linux'
    elif platform == "darwin":
        return 'mac'
    elif platform == "win32":
        return 'win'

################################################################################
################################################################################
def u_loadJson(file_name):
    with open(file_name) as f:
        data = json.load(f)
    return data

################################################################################
################################################################################
def u_fileList2array(file_name):
    print('Loading data from: ' + file_name)
    F = open(file_name,'r') 
    lst = []
    for item in F:
        item = item.replace('\\', '/').rstrip()
        lst.append(item)
    F.close()
    return lst

################################################################################
################################################################################
def u_fileList2array_(file_name):
    '''
    It loads data from file list which has the first row as the main root
    '''
    print('Loading data from: ' + file_name)
    F       = open(file_name,'r') 
    root    = F.readline().strip() 
    lst = []
    for item in F:
        #item = item.replace('\\', '/').rstrip()
        lst.append(item)
    F.close()
    return root, lst

################################################################################
################################################################################
def u_save2File(file_name, data):
    print('Saving data in: ' + file_name)
    F = open(file_name,'w') 
    F.write(data)
    F.close()

################################################################################
################################################################################
def u_saveList2File(file_name, data):
    print('Saving data in: ' + file_name)
    F = open(file_name,'w') 
    for item in data:
        item = item.strip()
        F.write(item + '\n')
    F.close()

################################################################################
################################################################################
def u_fileNumberList2array(file_name):
    print('Loading data from: ' + file_name)
    F = open(file_name,'r') 
    lst = []
    for item in F:
        if len(item) > 0:
            lst.append(float(item))
    F.close()
    return lst

################################################################################
################################################################################
def u_fileNumberMat2array(file_name):
    print('Loading data from: ' + file_name)
    F = open(file_name,'r') 
    lst = []
    for item in F:
        if len(item) > 0:
            item = item.split(' ')
            sub  = []
            for i in item:
                sub.append(float(i))
            lst.append(sub)
    F.close()
    return lst

################################################################################
################################################################################
''' save string matrix estructure into file'''
def u_fileString2DMat2array(file_name, token):
    print('Loading data from: ' + file_name)
    F = open(file_name,'r') 
    lst = []
    for item in F:
        if len(item) > 0:
            item = item.split(token)
            lst.append(tuple(item))
    F.close()
    return lst

################################################################################
################################################################################
def u_saveArray2File(file_name, data):
    print('Saving data in: ' + file_name)
    F = open(file_name,'w') 
    for item in data:
        F.write(str(item))
        F.write('\n')
    F.close()

################################################################################
################################################################################
def u_saveFlist2File(file_name, root, data):
    print('Saving data in: ' + file_name)
    F = open(file_name,'w') 
    F.write(root)
    for item in data:
        F.write('\n')
        F.write(str(item))
        
    F.close()

################################################################################
################################################################################
def u_saveArrayTuple2File(file_name, data):
    print('Saving data in: ' + file_name)
    F = open(file_name,'w') 
    for item in data:
        line = ''
        for tup in item:
            line += str(tup) + ' '
        F.write(line.strip())
        F.write('\n')
    F.close()
################################################################################
################################################################################
'''
Save dict into file, recommendably [.json]
'''
def u_saveDict2File(file_name, data):
    print ('Saving Dict data in: ', file_name)
    with open(file_name, 'w') as outfile:  
        json.dump(data, outfile)

################################################################################
################################################################################
def u_mkdir(directory):
    '''Crea un directorio si no existe'''
    if not os.path.exists(directory):
        print('Directorio creado en ', directory)
        os.makedirs(directory)

################################################################################
################################################################################
'''
it returns the complete file list in a list
'''
def u_listFileAll(directory, token):
    list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(token):
                list.append(root +'/'+ file)
    
    return sorted (list, key = u_stringSplitByNumbers)
    
################################################################################
################################################################################
'''
it returns a vector with separate root and files
'''
def u_listFileAllVec(directory, token, wdir = True):
    list = []
    for root, dirs, files in os.walk(directory):
        root        = root.replace('\\', '/') 
        sub_list    =[root.replace(directory, ''),[]]
        for file in files:
            if file.endswith(token) or file.endswith(token.upper()):
                if wdir:
                    file = root + '/' + file
                sub_list[1].append(file)
        if len(files) > 0:
            sub_list[1] = sorted (sub_list[1], key = u_stringSplitByNumbers)
            list.append(sub_list)

    return list

################################################################################
################################################################################
def u_listFileAllDic(directory, token=None):
    '''it returns a dictionary with complete map
    '''
    out     = {}
    files   = []
    if token != None:
        for entry in os.listdir(directory):
            local = os.path.join(directory, entry)
            if os.path.isdir(local):
                out[entry] = u_listFileAllDic(local, token)
            else:
                if entry.endswith(token) or \
                   entry.endswith(token.upper()):
                    files.append(entry)
    else:
        for entry in os.listdir(directory):
            local = os.path.join(directory, entry)
            if os.path.isdir(local):
                out[entry] = u_listFileAllDic(local, token)
            else:
                files.append(entry)
    
    if len(files):
        files   = sorted (files, key = u_stringSplitByNumbers)
        out['_files'] = files
    
    out['_path'] = directory

    return out

################################################################################
################################################################################
def u_listFileAllDic_2(base, base_local, token=None):
    '''it returns a dictionary with complete map, the path is separated
    '''
    out     = {}
    files   = []
    path    = base + base_local
    if token != None:
        for entry in os.listdir(path):
            local = u_joinPath([base + base_local, entry])
            if os.path.isdir(local):
                out[entry] = u_listFileAllDic_2(base, base_local+'/'+entry, token)
            else:
                if entry.endswith(token) or \
                   entry.endswith(token.upper()):
                    files.append(entry)
    else:
        for entry in os.listdir(path):
            local = u_joinPath([base + base_local, entry])
            if os.path.isdir(local):
                out[entry] = u_listFileAllDic_2(base, base_local+'/'+entry, token)
            else:
                files.append(entry)
    
    if len(files):
        files   = sorted (files, key = u_stringSplitByNumbers)
        out['_files'] = files
    
    out['_path'] = base_local

    return out


################################################################################
################################################################################
def u_getPath(file):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('inputpath', nargs='?', 
                        help='The input path. Default = conf.json')
    args = parser.parse_args()
    return args.inputpath if args.inputpath is not None else file

################################################################################
################################################################################
def u_loadFileManager(directive, token = ''):
    print(directive)
    if os.path.isfile(directive):
        file_list = []
        file = open(directive)
        for item in file:
            file_list.append(item)
    else:
        file_list   = u_listFileAll(directive, token)

    return sorted(file_list, key = u_stringSplitByNumbers)

################################################################################
################################################################################
''' console bar animation of process'''
def u_progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '>' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush() 
    
################################################################################
################################################################################
''' init a list with different list'''
def u_init_list_of_objects(size):
    list_of_objects = list()
    for i in range(0,size):
        list_of_objects.append( list() ) #different object reference each time
    return list_of_objects

################################################################################
################################################################################
def u_replaceStrList(str_list, token1, token2):
    ''' replace string in a list of strings'''
    for i in range(len(str_list)):
        str_list = str_list.replace(token1, token2)
    return str_list

################################################################################
################################################################################
''' split string by alfanumerical'''
def u_stringSplitByNumbers(x):
    r = re.compile('(\d+)')
    l = r.split(x)
    return [int(y) if y.isdigit() else y for y in l]

################################################################################
################################################################################
''' join string vector in only one string using a defined token'''
def u_fillVecToken(names, token = ' '):
    ans = names[0]
    for i in range(1, len(names)):
        if names[i] != '':
            ans += token + names[i] 
    return ans

################################################################################
################################################################################
''' os  similar to joinpath of python os'''
def u_joinPath(names):
    return u_fillVecToken(names, '/')

################################################################################
################################################################################
''' change into values for determinate key that contains token _pt '''
def u_look4PtInDict(dict_, root):
       for item in dict_:
           if item.find('_pt') > -1:
            dict_[item] = u_joinPath([root, dict_[item]])

################################################################################
################################################################################
def u_divideList(l, n):
    '''divide list into sublist o size n
    looping till length l
    reference variable
    yield acts like list generator
    use: a = list(u_divideList(l, n))
    '''
    for i in range(0, len(l), n):
        yield l[i:i + n]


