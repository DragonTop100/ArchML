import os
from icrawler.builtin import GoogleImageCrawler


def download_politicians(politicians, samples_per_person=100):
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '..',
                                             'data',
                                             'politicians'))
    for politician in politicians:
        save_path = os.path.join(base_path,
                                 'train',
                                 politician.lower().replace(' ', '_'))
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        print(f'Uploading images for {politician}')

        google_crawler = GoogleImageCrawler(storage={'root_dit': save_path})
        google_crawler.crawl(keyword=f'{politician} face portrait',
                             max_num=samples_per_person)

    print('All images have been downloaded')
