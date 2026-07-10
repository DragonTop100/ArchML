import os
from icrawler.builtin import BingImageCrawler


DEFAULT_TEMPLATES = [
    '{name} official portrait',
    '{name} press conference',
    '{name} official visit',
    '{name} speech',
    '{name} summit meeting',
]


def download_politicians(politicians, samples_per_person=150,
                         templates=None):
    templates = templates or DEFAULT_TEMPLATES
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '..',
                                             'data',
                                             'politicians',
                                             'raw'))
    per_query = max(1, samples_per_person // len(templates))
    for politician in politicians:
        name = politician.lower().replace(' ', '_')
        person_path = os.path.join(base_path, name)
        os.makedirs(person_path, exist_ok=True)

        print(f'Uploading images for {politician}')

        for template in templates:
            keyword = template.format(name=politician)
            crawler = BingImageCrawler(storage={'root_dir': person_path})
            crawler.crawl(
                keyword=keyword,
                max_num=per_query,
                file_idx_offset='auto'
            )

    print('All images have been downloaded')
