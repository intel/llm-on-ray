# PII removal for contact info

## Intro

PII removal for contact info is to replace personal information such as email, phone number to a random-non-sense string to protect personal infomation
This script is using multi processing method to speed up PPI removal

## Expected input and Output

Input format: a folder of *parquet, 'text' will required in parquet column names.

Out format: a folder of *parquet, 'text' will be processed and personal info will be replaced.

## How to RUN
```
conda create --name pyrecdp
conda activate pyrecdp
pip install pyrecdp --pre
pip install presidio_analyzer
python -m spacy download en_core_web_lg
python pii_redaction.py -d ../falcon-refinedweb -o ../falcon-refinedweb-pii_removal -mp 224
```

## NOTICE

We are running at file-wised parallism, usually a 300MB file took around 15-20min to complete, so you will see slow progress in progress bar.
One thing to identify the activity of the process may be using 'top' to check of there are multiple activitily running python processes.
