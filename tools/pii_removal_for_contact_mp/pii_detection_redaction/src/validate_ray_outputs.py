from datasets import load_dataset
# from presidio_analyzer import AnalyzerEngine
from utils import summarize_pii_entities
# import time
import json
# from redact_pii import redact_pii_with_random_values, redact_pii_with_tags
# from detect_pii import detect_other_piis, detect_phone_numbers, merge_outputs
import glob
import random
import os
import pandas as pd
# from matplotlib import pyplot as plt

import secrets

def write_merged_detect_redact_results_to_html(sample):
    # sample is a dictionary {'text': xxxx, pii: []}
    #text = sample['original']
    piis = sample['secrets']
    entity_categories = []
 
    if sample['modified']:
        #text_list = list(text)
        redacted_text = sample['text']
        redacted_text_list = list(redacted_text)
    
        
        n = 0
        for r in piis:
            start = int(r['start'])
            end = int(r['end'])
            # text_list.insert(start+3*n, '<mark><b>')
            # text_list.insert(end+1+3*n, "</b></mark>")
            # text_list.insert(end+2+3*n, '<b>[[['+r['type']+']]]</b>')
                    
            redacted_text_list.insert(start+3*n, '<mark><b>')
            redacted_text_list.insert(end+1+3*n, "</b></mark>")
            redacted_text_list.insert(end+2+3*n, '<b>[[['+r['type']+']]]</b>')

            n+=1
            entity_categories.append(r['type'])

    
        bolded = ''.join(redacted_text_list)
        #html = "<html>"+bolded+"</html>"
        #print(html)

        # redacted_marked = ''.join(redacted_text_list)
        summary = summarize_pii_entities(entity_categories)
        bolded = summary + '</p>'+ bolded

    else:
        bolded = sample['text']
        # redacted_marked = None

    return bolded, entity_categories




path = '/home/vmagent/app/falcon-refinedweb-pii-remove/'
datafile = glob.glob(path + '*.parquet')
# randomly pick one file from output
filename = secrets.choice(datafile)
output = 'pii_test'

# Check 1: load with pd to check schema and content
df = pd.read_parquet(filename)
print(df.head(10))

print(df.shape)


# Check 2: get statistics from a sample
def get_stats(row):
    count_dict = {
            'PHONE_NUMBER': 0,
            'IP_ADDRESS':0,
            'EMAIL': 0,
            'USER':0,
            'KEY':0
        }
     
    if row['modified'] == True:
        pii = row['secrets']
        num_piis = len(pii)
        for x in pii:
            count_dict[x['type']] += 1

    else:
        num_piis = 0

    return num_piis, count_dict


sample_files = random.sample(datafile, min(10, len(datafile)))
total_num_piis = 0
count_dict_all = {
        'PHONE_NUMBER': 0,
        'IP_ADDRESS':0,
        'EMAIL': 0,
        'USER':0,
        'KEY':0
    }

for f in sample_files:
    df = pd.read_parquet(f).sample(1000)
    for _, row in df.iterrows():
        num_piis, count_dict = get_stats(row)
        total_num_piis += num_piis
        for k, v in count_dict.items():
            count_dict_all[k] += v

print(count_dict_all)


# Check 3: visual check with html
df= df.sample(100)
html=""
num_piis = []
entities = []
# summary = 'Total number of samples: '+str(len(samples)) +'</p>'
summary = ""

for _, sample in df.iterrows():
    #print(sample['meta'])
    bolded, entity_categories = write_merged_detect_redact_results_to_html(sample)

    try:
        meta = sample['meta']
        html += (meta + '</p>'+bolded+"</p>")
    except:
        html += '</p>---------------------------</p>'
        html += '</p>'+bolded+"</p>"
    
    if sample['modified']:            
        # html += '</p>'+redacted+"</p>"
        num_piis.append(len(sample['secrets']))
        entities.extend(entity_categories)


assert sum(num_piis)==len(entities), 'number of entities not match'


summary += 'Total number of PIIs: {}'.format(len(entities))

summary += '</p>' + summarize_pii_entities(entities) +'</p>'

html = '<html>'+summary+html+"</html>"


output_path = path + 'validation/'
if not os.path.exists(output_path):
    os.mkdir(output_path)


output_file = output_path + '{}-pii-validation.html'.format(output)
f = open(output_file,"w")
f.write(html)
f.close()