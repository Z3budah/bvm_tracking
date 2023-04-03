# -*- coding: utf-8 -*-
"""bilibili.py
 Download baby video from bilibili.
"""
import logging
import os
import re
import json
import requests
import time
from bs4 import BeautifulSoup
from fake_useragent import UserAgent


class Bilibili:
    def __init__(self, keyword):
        ua = UserAgent()
        self.headers = {
            'User-Agent': ua.chrome,
            'referer': 'https://www.bilibili.com/'
        }
        self.keyword = keyword
        self.video_list = []

    def search_video(self, page):
        # video from bilibili, filter by: less than 10 mins +  order by pubdate +partition life parent-child
        url = (
                'https://search.bilibili.com/all?keyword=' + self.keyword + '&from_source=webtop_search&spm_id_from=333.1007&search_source=5&order=pubdate&duration=1&tids=254&page=' + str(
            page) + "&o=" + str((page - 1) * 36))
        # url = ('http://search.bilibili.com/all?keyword=' + self.keyword +       '&single_column=0&&order=dm&page='  + str(page))

        req = requests.get(url, headers=self.headers)
        logging.info("searching link {}".format(url))
        # soup = BeautifulSoup(req.text, "html.parser")
        # print(soup.prettify())
        content = req.text
        pattern = re.compile('<a href="//www.bilibili.com/video/(.*?)\?from=search"')
        # pattern = re.compile('bvid:"(.*?)",title')

        list_add = pattern.findall(content)
        # print(list_add)
        time.sleep(1)
        logging.info('第{}页'.format(page), list_add)
        self.get_video_url(list_add)

    def get_video_url(self, bv_list):
        for bv_id in bv_list:
            url = "https://www.bilibili.com/video/" + bv_id
            req = requests.get(url, headers=self.headers)

            video_info = re.findall(r'<script>window.__playinfo__=(.*?)</script>', req.text)[0]
            title = re.findall(r'<h1 title="(.*?)" class="video-title tit">', req.text)[0]

            video_info_data = json.loads(video_info)
            video_url = video_info_data['data']['dash']['video'][-1]['base_url']
            self.download_video(title, video_url)

    def download_video(self, title, video_url):
        req = requests.get(video_url, headers=self.headers)
        video_path = os.path.join(os.getcwd(), "bilibili", title + ".mp4")
        if req.status_code == 200:
            logging.info('file name: {}'.format(title))
            logging.info('file path: {}'.format(video_path))
            if os.path.isfile(video_path):
                logging.info('file already exists')
                return None
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
            # print(f'time cost:{cost_time:0.2f}s')
            # print(f'download speed:{file_size_MB / cost_time:0.2f}M/s')
