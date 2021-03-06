import argparse
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

LABELS = [
    'neutral',
    'happiness',
    'surprise',
    'sadness',
    'anger',
    'disgust',
    'fear',
    'contempt',
    'unknown',
    'NF',
]


def unpack(base_folder: str, fer_path: str, ferplus_path: str) -> None:
    folder_aliases = {
        'Training': 'train',
        'PublicTest': 'val',
        'PrivateTest': 'test',
    }

    dataframe = pd.concat(
        [
            pd.read_csv(ferplus_path),
            pd.read_csv(fer_path, usecols=['pixels']),
        ],
        axis=1,
    ).dropna()

    for key, value in folder_aliases.items():
        subset = dataframe[dataframe['Usage'] == key]
        subset = subset.rename(columns={'Image name': 'filename'})
        subset = subset[['filename', *LABELS]]
        subset.to_csv(os.path.join(base_folder, '..', f'{value}.csv'), index=False)

    os.mkdir(base_folder)

    for folder in folder_aliases.values():
        os.mkdir(os.path.join(base_folder, folder))

    for usage, image_name, pixels in dataframe[['Usage', 'Image name', 'pixels']].values:
        image = Image.fromarray(np.fromstring(pixels, np.uint8, 48 * 48, ' ').reshape(48, 48))
        image_path = os.path.join(base_folder, folder_aliases[usage], image_name)
        image.save(image_path, compress_level=0)


def majority(file_path: str) -> str:
    dataframe = pd.read_csv(file_path)
    save_path = os.path.join(os.path.dirname(sys.argv[0]), f'majority_{os.path.basename(file_path)}')

    emotion_votes = dataframe[[*LABELS]]
    emotion_votes = emotion_votes.replace(1, 0)

    labels = []

    for line in emotion_votes.values:
        if np.max(line) > 0.5 * np.sum(line):
            labels.append(LABELS[np.argmax(line)])
        else:
            labels.append(np.nan)

    dataframe['class'] = labels
    dataframe = dataframe.dropna()
    dataframe = dataframe[(dataframe['class'] != 'unknown') & (dataframe['class'] != 'NF')]
    dataframe[['filename', 'class']].to_csv(save_path, index=False)

    return save_path


def info(file_path: str) -> None:
    dataframe = pd.read_csv(file_path)

    total = len(dataframe.index)

    for label in LABELS:
        n = len(dataframe[dataframe['class'] == label].index)
        print(f'\t{label}: {n} ({n / total * 100:.2f}%)')

    print(f'\n\ttotal: {total}')


def balance(file_path: str, max_images_per_class: int) -> str:
    dataframe = pd.read_csv(file_path)
    save_path = os.path.join(os.path.dirname(sys.argv[0]), f'balanced_{os.path.basename(file_path)}')

    chunks = []
    for label in LABELS[:-2]:
        class_elements = dataframe[dataframe['class'] == label]
        n = len(class_elements.index)

        if n >= max_images_per_class:
            chunks.append(class_elements.sample(max_images_per_class))
        else:
            chunks.append(pd.concat([class_elements] * round(max_images_per_class / n)))

    dataframe = pd.concat(chunks)
    dataframe = dataframe.sort_values(by='filename')
    dataframe.to_csv(save_path, index=False)

    return save_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_unpack = subparsers.add_parser('unpack')
    parser_unpack.add_argument('-d', '--base_folder', type=str, default='data/images')
    parser_unpack.add_argument('-fer', '--fer_path', type=str, default='data/fer2013.csv')
    parser_unpack.add_argument('-ferplus', '--ferplus_path', type=str, default='data/fer2013new.csv')

    parser_majority = subparsers.add_parser('majority')
    parser_majority.add_argument('-f', '--file_path', type=str, required=True)

    parser_info = subparsers.add_parser('info')
    parser_info.add_argument('-f', '--file_path', type=str, required=True)

    parser_balance = subparsers.add_parser('balance')
    parser_balance.add_argument('-f', '--file_path', type=str, required=True)
    parser_balance.add_argument('-n', '--max_images_per_class', type=int)

    args = parser.parse_args()

    if args.mode == 'unpack':
        unpack(args.base_folder, args.fer_path, args.ferplus_path)

    elif args.mode == 'majority':
        print('Info:')
        info(majority(args.file_path))

    elif args.mode == 'info':
        print('Info:')
        info(args.file_path)

    elif args.mode == 'balance':
        print('Before:')
        info(args.file_path)
        print('\nAfter:')
        info(balance(args.file_path, args.max_images_per_class))
