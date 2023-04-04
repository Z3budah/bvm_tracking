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
    scrape_bilibili()
