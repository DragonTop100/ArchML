import os
import random
import shutil
from icrawler.builtin import BingImageCrawler


def download_politicians(politicians, samples_per_person=100, val_ratio=0.2):
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '..',
                                             'data',
                                             'politicians'))
    for politician in politicians:
        name = politician.lower().replace(' ', '_')
        train_path = os.path.join(base_path, 'train', name)
        val_path = os.path.join(base_path, 'val', name)

        os.makedirs(train_path, exist_ok=True)
        os.makedirs(val_path, exist_ok=True)

        print(f'Uploading images for {politician}')

        google_crawler = BingImageCrawler(storage={'root_dir': train_path})
        google_crawler.crawl(keyword=f'{politician} face portrait',
                             max_num=samples_per_person)

        images = [
                f for f in os.listdir(train_path)
                if os.path.isfile(os.path.join(train_path, f))
        ]

        val_count = int(len(images) * val_ratio)
        to_move = random.sample(images, val_count)

        for filename in to_move:
            train = os.path.join(train_path, filename)
            val = os.path.join(val_path, filename)
            shutil.move(train, val)

    print('All images have been downloaded')
