import os
import sys
from zipfile import ZipFile

from acpsr.download import fetch_content


if not os.path.exists('./data'):
    sys.stderr.write('create `data` directory!')
    os.mkdir('./data')

if not os.path.exists('./data/acupoint'):
    fetch_content('acupoint.zip', './data')
    sys.stderr.write('unzip files, it can take a while.')
    with ZipFile('./data/acupoint.zip', 'r') as zipfile:
        zipfile.extractall('./data')

gen_pres_path = os.path.join('./data', 'combined_prescription')
if not os.path.exists(gen_pres_path):
   sys.stderr.write('create'+gen_pres_path+'directory!') 
   os.mkdir(gen_pres_path)
if not os.path.exists(os.path.join(gen_pres_path,'Cleaned.txt')):
    fetch_content('Cleaned.txt', gen_pres_path)
