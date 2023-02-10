from wget import download
import os
import sys

def fetch_content(file_name, save_path):
    sys.stderr.write('Start downloading '+file_name)
    try:
        response = download('https://www.goldfishl.me/downloads/'+file_name, os.path.join(save_path,file_name))
    except HTTPError:
       sys.stderr.write("Website or content is not available, you can send an email to sicnu.long@gmail.com, or contact me directly") 
    sys.stderr.write('Download completed')