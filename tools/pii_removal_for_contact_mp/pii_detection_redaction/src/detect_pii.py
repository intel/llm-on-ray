
#from presidio_analyzer import AnalyzerEngine
from utils import parse_recognizer_result, high_risk_tags
from bigscience_pii_detect_redact import matches_date_pattern, detect_pii



def detect_phone_numbers(text, analyzer):
    # use presidio phone recognizer to detect phone numbers
    # threshold is set to 0.4 based on a sample study
    results = analyzer.analyze(text=text,
                            entities=['PHONE_NUMBER'],
                            #language='en',
                            #score_threshold=0.4,
                            #return_decision_process=True
                            )
    
    pii_list = []

    if len(results)>0:
        for result in results:
            # parse the output into dictionary
            pii_dict = parse_recognizer_result(result)

            # check if the number string is a date
            number_str = text[pii_dict['start']: pii_dict['end']]

            if matches_date_pattern(number_str):
                #print('Date, not phone number')
                pass
        
            else:
                pii_dict['value']=number_str
                pii_list.append(pii_dict)
                #print(pii_dict)

    return pii_list



def detect_other_piis(text):
    matches = detect_pii(text, None, high_risk_tags)
    if len(matches)>0:
        pii_list = []
        for m in matches:
            pii = {}
            pii['type']=m[-2]
            pii['start']=m[1][0]
            pii['end']=m[1][1]
            pii['value']=m[0]
            #print(pii)
            pii_list.append(pii)
        
        return pii_list
    else:
        return None
    
def merge_outputs(presidio_outputs, bigscience_outputs): 
    if bigscience_outputs!=None:
        piis = presidio_outputs + bigscience_outputs
        # TODO: sometimes KEY and PHONE_NUMBER overlap
        # when merging, only keep one of them
        # right now, the short-cut is to have the KEY and PHONE_NUMBER replacement to be the same format
    
        # detected_spans = []
        # piis_to_remove = []
        # for pii in piis:
        #     span = (pii['start'], pii['end'])
        #     if span in detected_spans:
        #         #remove pii from piis
        #         print('remove this pii: ', pii)
        #         piis_to_remove.append(pii)
            
        #     detected_spans.append(span)

        # piis = [pii for pii in piis if pii not in piis_to_remove]

    else:
        piis = presidio_outputs
    return piis
