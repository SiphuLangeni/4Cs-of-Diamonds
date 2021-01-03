import requests
import pandas as pd


class Diamonds:
    
    def __init__(self, page_num=1, min_carat=0.25, max_carat=2):

        self.page_num = page_num
        self.min_carat = min_carat
        self.max_carat = max_carat
        
        self.url = 'https://www.brilliantearth.com/loose-diamonds/list/'
        self.num_records = 0
        
        self.diamond_id = []
        self.carat = []
        self.color = []
        self.clarity = []
        self.cut = []
        self.price = []

    
    def get_diamonds(self):
        
        path = f'?shapes=Round&min_carat={self.min_carat}&max_carat={self.min_carat}&page={self.page_num}'
        full_path = self.url + path
        
        response = requests.get(full_path).json()

        for diamond in response['diamonds']:
            self.diamond_id.append(diamond['id'])
            self.carat.append(diamond['carat'])
            self.color.append(diamond['color'])
            self.clarity.append(diamond['clarity'])
            self.cut.append(diamond['cut'])
            self.price.append(diamond['price'])
            self.num_records += 1

        if len(response['diamonds']) == 20:
            self.page_num += 1
            self.get_diamonds()
            
        else:
            print(f'***** Retrived {self.num_records:,} diamonds of {self.min_carat:.2f} carat weight *****')

        if self.min_carat < self.max_carat:
            self.min_carat += 0.01
            self.page_num = 1
            self.get_diamonds()


    def create_frame(self):
        
        data = {
            'diamond_id': self.diamond_id,
            'carat': self.carat,
            'color': self.color,
            'clarity':self.clarity,
            'cut': self.cut,
            'price': self.price
        }
        
        return pd.DataFrame(data)
    