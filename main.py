# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import logging
import sys
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from bilibili import Bilibili

import requests
import os
import re
import json
import time
from fake_useragent import UserAgent

TOTAL_PAGE = 28


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
# logging.getLogger().setLevel(logging.INFO)

def scrape_bilibili():
    pages = range(1, TOTAL_PAGE)
    bilibili = Bilibili('婴儿')
    with Pool(processes=4) as pool:
        result = list(tqdm(pool.imap(bilibili.search_video, pages), total=TOTAL_PAGE))


if __name__ == '__main__':
    print_hi('PyCharm')
    bv_id = "BV1dm4y1A7j3"
    ua = UserAgent()
    url = "https://www.bilibili.com/video/" + bv_id
    req = requests.get(url, headers={
            'User-Agent': ua.chrome,
            'referer': 'https://www.bilibili.com/'
        })

    video_info = re.findall(r'<script>window.__playinfo__=(.*?)</script>', req.text)[0]
    title = re.findall(r'<h1 title="(.*?)" class="video-title tit">', req.text)[0]

    video_info_data = json.loads(video_info)

    print(json.dumps(video_info_data, indent=2))
    video_url = video_info_data['data']['dash']['video'][-1]['base_url']

    req = requests.get(video_url, headers={
            'User-Agent': ua.chrome,
            'referer': 'https://www.bilibili.com/'
        })
    video_path = os.path.join(os.getcwd(), "bilibili", title + ".mp4")
    if req.status_code == 200:
        logging.info('file name: {}'.format(title))
        logging.info('file path: {}'.format(video_path))
        if os.path.isfile(video_path):
            logging.info('file already exists')
            sys.exit()
        chunk_size = 1024
        file_size = int(req.headers['content-length'])

        done_size = 0

        file_size_MB = file_size / 1024 / 1024
        logging.info('file size: %s MB', file_size_MB)
        # print(f"file size：{file_size_MB:0.2f}MB")
        start_time = time.time()
        try:
            with open(f"{video_path}", mode='wb') as f:
                for chunk in req.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    done_size += len(chunk)
                    logging.info('download %s %', done_size / file_size * 100)
                    # print(f'\r download:{done_size / file_size * 100:0.2f}%', end='')
        except Exception as e:
            logging.error("Failed to download video:{}".format(e))
            logging.error('error occurred while downloading {}'.format(video_path))
        end_time = time.time()
        cost_time = end_time - start_time
        logging.info('time cost: %s', cost_time)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
