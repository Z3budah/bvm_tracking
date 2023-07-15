# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

from video_scrape.bilibili import Bilibili
from video_scrape.youtube import Yt_dl

import os
import re

TOTAL_PAGE = 28

# Press the green button in the gutter to run the script.
# logging.getLogger().setLevel(logging.INFO)

def scrape_bilibili():
    pages = range(1, TOTAL_PAGE)
    bilibili = Bilibili('婴儿')
    with Pool(processes=4) as pool:
        result = list(tqdm(pool.imap(bilibili.search_video, pages), total=TOTAL_PAGE))

def scrape_youtube():
    yt_dl = Yt_dl()
    yt_dl.search_video("fidgety+movements+infant", 20)
    yt_dl.download_video()

if __name__ == '__main__':
    scrape_bilibili()
    scrape_youtube()
