from wget import download
import os
import sys

def fetch_content(file_name, save_path):
    print('Start downloading', file_name, '->', save_path)
    try:
        response = download('https://www.goldfishl.me/downloads/'+file_name, os.path.join(save_path,file_name))
    except HTTPError:
       print("Website or content is not available, you can send an email to sicnu.long@gmail.com, or contact me directly") 
    print('Download completed')