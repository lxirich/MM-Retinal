import json
import pandas as pd

with open('./FFA-IR/1.0.0/ffair_annotation.json', 'r') as file:
    ann = json.load(file)

dataframe = pd.DataFrame()


for key, value in ann.items():
    for k, v in value.items():
        dic = dict()
        dic['Split'] = key
        dic['name'] = k
        for ks, vs in v.items():
            dic[ks] = vs
        dataframe = pd.concat([dataframe, pd.DataFrame([dic])], ignore_index=True)

dataframe.to_csv('./fair_annotation.csv', index=False, encoding='utf-8-sig')

