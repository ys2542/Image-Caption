import json

with open('c.json') as json_data:
    d = json.load(json_data)
    
f = open('caption.txt','w')

for i in range(len(d['annotations'])):
    for j in range(len(d['images'])):
        if d['annotations'][i]['image_id'] == d['images'][j]['id']:
            print(d['images'][j]['file_name'], d['annotations'][i]['caption'], file = f)
