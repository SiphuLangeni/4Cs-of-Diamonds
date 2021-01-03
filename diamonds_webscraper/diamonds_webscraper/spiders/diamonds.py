# -*- coding: utf-8 -*-
import scrapy
import json
import pandas


class DiamondsSpider(scrapy.Spider):

    def __init__(self):

        self.page_num = page_num
        self.min_carat = min_carat
        self.max_carat = max_carat

    name = 'diamonds'
    allowed_domains = ['brilliantearth.com/loose-diamonds/list/']
    start_urls = [
        f'brilliantearth.com/loose-diamonds/list/?shapes=Round&min_carat={self.min_carat}&max_carat={self.max_carat}&page={self.page_num}'
    ]


    def parse(self, response):
        data = json.loads(response.text)
        for diamond in data['diamonds']:
            yield {
                'diamond_id': diamond['id'],
                'carat': diamond['carat'],
                'color': diamond['color'],
                'clarity': diamond['clarity'],
                'cut': diamond['cut'],
                'price': diamond['price']
            }
        
        total_pages = ceil(data['total_count'] / 20)
        next_page_number = data['page'] + 1
        full_path = data['path']
        min_carat = re.findall('(?<=min_carat=)\w+\.\w+', full_path.lower())[0]
        max_carat = re.findall('(?<=max_carat=)\w+\.\w+', full_path.lower())[0]
        next_min_carat = float(min_carat) + 0.1
        next_max_carat = float(max_carat) + 0.1
    
        if total_pages >= next_page_number:
            next_page_url = f'https://www.brilliantearth.com/loose-diamonds/list/?shapes=Round&min_carat={min_carat}&max_carat={max_carat}&page={next_page_number}'
        else:
            next_page_url = f'https://www.brilliantearth.com/loose-diamonds/list/?shapes=Round&min_carat={next_min_carat:.2f}&max_carat={next_max_carat:.2f}&page=1'
        
        yield scrapy.Request(next_page_url, callback=self.parse)
