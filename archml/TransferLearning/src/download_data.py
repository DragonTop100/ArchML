import os
from icrawler.builtin import BingImageCrawler


def download_politicians(politicians, samples_per_person=150):
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '..',
                                             'data',
                                             'politicians',
                                             'raw'))
    for politician in politicians:
        name = politician.lower().replace(' ', '_')
        person_path = os.path.join(base_path, name)
        os.makedirs(person_path, exist_ok=True)

        print(f'Uploading images for {politician}')

        crawler = BingImageCrawler(storage={'root_dir': person_path})
        crawler.crawl(
                keyword=f'{politician} official portrait face close up',
                max_num=samples_per_person,
                filters={'size': 'large'}
        )

    print('All images have been downloaded')
