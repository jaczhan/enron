import json
import os
import shutil
from os import listdir
from os.path import isfile, join


folder =  "c:/Users/jacks/Documents/INTA 6450 Data Analytics and Security/Project/group/strik_data_california"
ranged_folder = "c:/Users/jacks/Documents/INTA 6450 Data Analytics and Security/Project/group/ranged_data"
mmip_folder = "c:/Users/jacks/Documents/INTA 6450 Data Analytics and Security/Project/group/mmip_data"
#main_keywords = ["california", "power", "energy", "state", "price", "electricity", "plant", "market", "electric", "rate", "customer", "bill", "direct", "cost", "access", "utility", "issue", "exchange"]
main_keywords = ["MMIP"]


def  keywords_filter(data, keywords):
    for i in keywords:
        if i.lower() in data.lower() and "california" in data.lower():
            return True
    return False


def filter_files(mypath, destination):
    files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    for file in files:
        full_path = os.path.join(mypath, file)
        f = open(full_path)
        data = json.load(f)

        raw_text = data['text']
        
        if keywords_filter(raw_text, main_keywords):
            shutil.copy2(full_path, destination)


def write_to_file(file_name, data, folder):
    path = os.path.join(folder, file_name)
    path += ".txt"
    with open(path, 'w') as outfile:
        json.dump(data, outfile)



if __name__ == '__main__':
    filter_files(ranged_folder, mmip_folder)