import json
from datasets import load_dataset
import random
from bigscience_pii_detect_redact import detect_pii

piis_not_to_consider =['PERSON',
                       'NRP',
                       'LOCATION',
                       'DATE_TIME',
                       'URL'
                       ]

def write_to_html(result, text_corpus):
    idx = result['doc_idx']
    text = text_corpus[idx]['text']
    text_list = list(text)
    
    entity_categories = []
    n = 0
    for r in result['pii']:
        start = int(r['start'])
        end = int(r['end'])
        text_list.insert(start+3*n, '<mark><b>')
        text_list.insert(end+1+3*n, "</b></mark>")
        text_list.insert(end+2+3*n, '<b>[[['+r['type']+']]]</b>')
        n+=1
        entity_categories.append(r['type'])
    
    bolded = ''.join(text_list)
    #html = "<html>"+bolded+"</html>"
    #print(html)

    summary = summarize_pii_entities(entity_categories)
    
    bolded = summary + '</p>'+ bolded
    return bolded, entity_categories


def summarize_pii_entities(entity_categories):
    unique_categories = list(set(entity_categories))
    summary = 'PIIs: '
    for e in unique_categories:
        occurences = entity_categories.count(e)
        summary += (e + ": "+str(occurences)+'; ')
    return summary

def parse_recognizer_result(result):
    #temp = result.split(',')
    #assert len(temp)==4, 'a valid result should have 4 fields, but only got {} fields'.format(len(temp))
    parsed_dict = {}
    
    parsed_dict['type']=result.entity_type
    parsed_dict['start']=result.start
    parsed_dict['end']=result.end #temp[2][5:]
    parsed_dict['score']=result.score #temp[3][6:]
    
    return parsed_dict


def count_num_piis_to_consider(result):
    #result is a dictionary of this format
    # {doc_idx:123, num_pii: 2, pii: [{type: ABC, start:234, end: 256, score:0.6}, {}]}
    filtered_piis = []
    piis = result['pii']
    num_piis_to_consider = 0
    for pii in piis:
        if pii['type'] in piis_not_to_consider:
            #print('Not including {} category'.format(pii['type']))
            continue
        else:
            num_piis_to_consider += 1
            #print(pii)
            if pii['type'] != 'IP_ADDRESS' and pii['type'] !='EMAIL_ADDRESS':
                pii['type'] = 'ID_NUM_STR'
            #print(pii)
            filtered_piis.append(pii)
    print('number of piis to consider: ', num_piis_to_consider)
    print('filtered piis: ',filtered_piis)

    return num_piis_to_consider, filtered_piis

def filter_results_by_category(results):
    filtered_results = []
    for result in results:
        num_piis_to_consider, filtered_piis = count_num_piis_to_consider(result)
        if  num_piis_to_consider>0:
            result['pii'] = filtered_piis
            result['num_pii']=len(filtered_piis)
            filtered_results.append(result)
    print('filtered results: ',filtered_results)
    return filtered_results

def sample_results(results, number_of_samples):
    random.seed(1234)
    if len(results) > number_of_samples:
        return random.sample(results, number_of_samples)
    else:
        return results

# this tag list is copied from 
# https://github.com/bigscience-workshop/data-preparation/blob/main/preprocessing/training/02_pii/bigscience_pii_detect_redact.py#L53
high_risk_tags = {'KEY', 'EMAIL', 'USER', 'IP_ADDRESS'} # , 'NUMBER', "ID"}
    
def detect_with_bigscience_pii_single_sample(text):
    matches = detect_pii(text, None, high_risk_tags)
    if len(matches)>0:
        pii_list = []
        for m in matches:
            print(m)
            pii = {}
            pii['type']=m[-2]
            pii['start']=m[1][0]
            pii['end']=m[1][1]
            print(pii)
            pii_list.append(pii)
        
        return pii_list
    else:
        return None
    
def is_phone_number(matched_str):
    DEFAULT_SUPPORTED_REGIONS = ("US", "UK", "DE", "FE", "IL", "IN", "CA", "BR")
    #valid = phonenumbers.is_valid_number(matched_str)
    print(matched_str)
    for region in DEFAULT_SUPPORTED_REGIONS:
        try:
            parsed_number = phonenumbers.parse(matched_str)
        except:
            print('cannot parse the string as phone number')
            return False
        
        flag = phonenumbers.is_possible_number(parsed_number)
        if flag == True:
            print('KEY is PHONE_NUMBER')
            return True
            
    return False
    


def remove_phone_numbers_from_bigscience_results(matches):
    # use bigscience-pii to detect
    # emails, ip addresses, usernames, id alphanumerics
    if len(matches)>0:
        pii_list = []
        phone_matches = []
        for i, m in enumerate(matches):
            matched_str = m[0]
            if is_phone_number(matched_str):
                phone_matches.append(i)
            # else:
            #     # print(m)
            #     pii = {}
            #     pii['type']=m[-2]
            #     pii['start']=m[1][0]
            #     pii['end']=m[1][1]
            #     print(pii)
            #     pii_list.append(pii)

        
        matches = [matches[i] for i in range(len(matches)) if i not in phone_matches]   
    return matches
