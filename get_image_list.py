import os
import glob
import csv

from tqdm import tqdm


def get_dollarstreet_image_list():
    data_root = '/private/home/changhan/projects/image_classification_fairness/data'
    image_root = os.path.join(data_root, 'dollarstreet')
    output_path = os.path.join(data_root, 'dollarstreet_img_list.txt')

    file_paths = sorted(os.listdir(image_root))
    count = 0
    with open(output_path, 'w') as f:
        for d in tqdm(file_paths):
            path = os.path.join(image_root, d)
            if not os.path.isdir(path):
                continue
            for p in glob.glob(os.path.join(path, '*.jpg')):
                img_id = os.path.basename(p).replace('.jpg', '')
                f.write(f'{p}\t{img_id}\n')
                count += 1
    print(f'{count} images in total')


def get_oix_image_list():
    # image_root = '/private/home/changhan/data/datasets/oix'
    image_root = '/scratch/changhan/data/oix'
    output_path = '/private/home/changhan/projects/image_classification_fairness/data/oix_img_list.txt'

    img_id_to_part_id = {}
    for i in tqdm(range(10)):
        for p in tqdm(os.listdir(os.path.join(image_root, f'crowdsource_images-0000{i}-of-00010'))):
            if p.endswith('.jpg'):
                img_id_to_part_id[p.replace('.jpg', '')] = i
    missing = set()
    count = 0
    with open(os.path.join(image_root, 'extended-crowdsourced-image-ids.csv')) as f_in:
        reader = csv.DictReader(f_in, delimiter=',')
        with open(output_path, 'w') as f_out:
            for r in tqdm(reader):
                img_id = r['ImageID']
                if img_id not in img_id_to_part_id:
                    missing.add(img_id)
                    continue
                path = os.path.join(image_root, f'crowdsource_images-0000{img_id_to_part_id[img_id]}-of-00010',
                                    img_id + '.jpg')
                f_out.write(f'{path}\t{img_id}\n')
                count += 1
    print('missing {} images ({})'.format(len(missing), ','.join(missing)))


def get_openimages_val_image_list():
    image_root = '/datasets01/open_images/092418/val'
    paths = [os.path.join(image_root, p) for p in tqdm(os.listdir(image_root)) if p.endswith('.jpg')]
    assert len(paths) == 41620

    output_path = '/private/home/changhan/projects/image_classification_fairness/data/openimages_val_img_list.txt'
    count = 0
    with open(output_path, 'w') as f:
        for p in paths:
            img_id = os.path.basename(p).replace('.jpg', '')
            f.write(f'{p}\t{img_id}\n')
            count += 1
    print(f'{count} images in total')


if __name__ == '__main__':
    get_oix_image_list()
