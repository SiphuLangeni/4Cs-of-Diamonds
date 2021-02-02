import requests
import csv
import logging


logging.basicConfig(format='%(message)s', level=logging.INFO)
logger = logging.getLogger()


class Diamonds:
    
    def __init__(self, page_num=1, min_carat=0.25, max_carat=13.0):

        self.page_num = page_num
        self.min_carat = min_carat
        self.max_carat = max_carat
        self.current_carat = self.min_carat
        
        self.url = 'https://www.brilliantearth.com/loose-diamonds/list/'
        self.current_records = 0
        self.total_records = 0
        
        self.upc = []
        self.cut = []
        self.colour = []
        self.clarity = []
        self.carat = []
        self.x = []
        self.y = []
        self.z = []
        self.lw_ratio = []
        self.depth = []
        self.table = []
        self.girdle = []
        self.culet = []
        self.fluorescence = []
        self.polish = []
        self.symmetry = []
        self.report = []
        self.origin = []
        self.certificate_number = []
        self.price = []

    
    def get_diamonds(self):
        
        path = f'?shapes=Round&min_carat={self.current_carat}&max_carat={self.current_carat}&page={self.page_num}'
        full_path = self.url + path
        
        response = requests.get(full_path).json()
        
        if bool(response) == False:
            logger.info(
                f'***** Retrived {self.current_records:,} '
                f'diamonds of {self.current_carat:.2f} carat weight *****'
            )
            
            if self.current_carat < self.max_carat:
                self.current_carat += 0.01
                self.get_diamonds()
                
            else:
                logger.info(
                    f'***** Retrived {self.total_records:,} '
                    f'diamonds of {self.min_carat:.2f} - {self.max_carat:.2f} carat weight *****'
                )
        
        else:
            for diamond in response['diamonds']:
                self.upc.append(diamond['upc'])
                self.cut.append(diamond['cut'])
                self.colour.append(diamond['color'])
                self.clarity.append(diamond['clarity'])
                self.carat.append(diamond['carat'])
                measurements = diamond['measurements'].split(' x ')
                x = float(measurements[0])
                y = float(measurements[1])
                z = float(measurements[2])
                lw_ratio = round(x / y, 4)
                self.x.append(x)
                self.y.append(y)
                self.z.append(z)
                self.lw_ratio.append(lw_ratio)
                self.depth.append(diamond['depth'])
                self.table.append(diamond['table'])
                self.girdle.append(diamond['girdle'])
                self.culet.append(diamond['culet'])
                self.fluorescence.append(diamond['fluorescence'])
                self.polish.append(diamond['polish'])
                self.symmetry.append(diamond['symmetry'])
                self.report.append(diamond['report'])
                self.origin.append(diamond['origin'])
                self.certificate_number.append(diamond['certificate_number'])
                self.price.append(diamond['price'])
                self.current_records += 1
                self.total_records += 1

            if response['total_count'] % 20 == 0:

                while self.page_num < (response['total_count'] / 20):
                    self.page_num += 1
                    self.get_diamonds()

                logger.info(
                    f'***** Retrived {self.current_records:,} '
                    f'diamonds of {self.current_carat:.2f} carat weight *****'
                )
                self.current_records = 0

            else:

                if len(response['diamonds']) == 20:
                    self.page_num += 1
                    self.get_diamonds()

                else:
                    logger.info(
                        f'***** Retrived {self.current_records:,} '
                        f'diamonds of {self.current_carat:.2f} carat weight *****'
                    )
                    self.current_records = 0

            if self.current_carat < self.max_carat:
                self.current_carat += 0.01
                self.page_num = 1
                self.get_diamonds()

    
    
            