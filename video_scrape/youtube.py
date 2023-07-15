# -*- coding: utf-8 -*-
"""youtube.py
 Crawl infant videos from youtube.
"""

from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver import ChromeOptions
# Chrome - Selenium 3
from webdriver_manager.chrome import ChromeDriverManager

import time
import re
import os
import logging
from pytube import YouTube
from tqdm import tqdm


class Yt_dl:
    def __init__(self):
        option = ChromeOptions()
        option.add_experimental_option('excludeSwitches', ['enable-automation'])
        option.add_experimental_option('useAutomationExtension', False)
        option.add_argument('--headless')

        # update webdriver chrome
        self.browser = webdriver.Chrome(ChromeDriverManager().install(), options=option)
        self.browser.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
            'source': 'Object.defineProperty(navigator, "webdriver",{get: () => undefined})'
        })

        self.video_list = []

    def search_video(self, keyword, video_num):
        msgtimeout = 1
        try:
            # video from youtube with keyword
            url = 'https://www.youtube.com/results?search_query=' + re.sub(" ","+",keyword)
            # url = 'https://www.youtube.com/results?search_query=fidgety+movements+infant'

            self.browser.get(url)

            time.sleep(2)
            temp_height = 0
            video_url = []

            while len(video_url) < video_num:
                # scroll down the bar
                # browser.execute_script("window.scrollBy(0,5000)")
                # wait to get video_id
                time.sleep(5)
                video_id = self.browser.find_elements(By.ID, 'video-title')
                cur_len = len(video_url)
                for i, vid in enumerate(video_id):
                    if i >= cur_len:
                        v_url = vid.get_attribute('href')
                        if v_url is not None:
                            video_url.append(v_url)

                # get the distance from the current scroll bar to the top
                check_height = self.browser.execute_script(
                    "return document.documentElement.scrollTop || window.pageYOffset || document.body.scrollTop;")
                if check_height == temp_height:
                    break
                temp_height = check_height

            msgtimeout = 0
            self.video_list = video_url
            logging.info("Search {} videos according to {}.".format(len(video_url),self.keyword))

        except TimeoutException as e:
            logging.error(type(e))
            logging.error("Fail to search video: {}".format(e))
            self.browser.delete_all_cookies()
            logging.info("Clear cookies, revisit {}".format(url))
            msgtimeout = 0
            return search_video(self, keyword, video_num)
        except Exception as e:
            logging.info(type(e))
            logging.error("Fail to search video: {}".format(e))
            msgtimeout = 0
        finally:
            if msgtimeout:
                logging.error("Timeout Exception")

    def download_video(self, video_dir='../youtube'):
        os.chdir(video_dir)
        for video_url in tqdm(self.video_list):
            try:
                yt = YouTube(video_url)
                logging.info('download... {}'.format(yt.title))
                yt.streams.filter().get_lowest_resolution().download(filename=(yt.video_id + '.mp4'))
            except Exception as e:
                logging.error("Fail to download video: {}".format(e))


