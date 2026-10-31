import argparse
import csv
import hashlib
import random
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def selected_dir(desktop):
    return desktop / 'BOSSbase_selected_500'


def generated_dir(desktop):
    return desktop / 'BOSSbase_generated_500'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select(desktop):
    """Sample 500 distinct PGM files with seed 42, leaving BOSSbase intact."""
    source = desktop / 'BOSSbase_1.01'
    images = sorted(source.glob('*.pgm'))
    assert len(images) >= 500, f'Found only {len(images)} PGM files'
    selected = random.Random(42).sample(images, 500)
    destination = selected_dir(desktop)
    destination.mkdir(exist_ok=False)
    for image in selected:
        shutil.copy2(image, destination / image.name)
    print(f'Selected {len(selected)} images')
    print(f'Saved copies in: {destination}')


def hashes(desktop):
    """Record SHA-256 values for the selected source images."""
    folder = selected_dir(desktop)
    images = sorted(folder.glob('*.pgm'))
    assert len(images) == 500, f'Expected 500 images, found {len(images)}'
    manifest = folder / 'source_manifest.csv'
    with manifest.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'sha256'])
        for image in images:
            writer.writerow([image.name, digest(image)])
    print(f'Recorded hashes for {len(images)} images')
    print(f'Manifest: {manifest}')


def generate(desktop):
    """Generate random-position LSB copies at 0.1 and 0.4 bpp."""
    from PIL import Image
    sources = selected_dir(desktop)
    root = generated_dir(desktop)
    root.mkdir(exist_ok=True)
    images = sorted(sources.glob('*.pgm'))
    assert len(images) == 500, f'Expected 500 originals, found {len(images)}'
    manifest = root / 'stego_manifest.csv'
    with manifest.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['source_filename', 'payload_bpp', 'seed', 'bits_embedded',
                         'pixels_changed', 'source_sha256', 'stego_sha256'])
        for number, source in enumerate(images, start=1):
            source_hash = digest(source)
            with Image.open(source) as image:
                assert image.mode == 'L' and image.size == (512, 512)
                original_pixels = list(image.getdata())
                size = image.size
            for payload, folder_name in [(0.1, '0p1'), (0.4, '0p4')]:
                folder = root / folder_name
                folder.mkdir(exist_ok=True)
                output = folder / source.name
                seed_text = f'lsb-v1:{source.name}:{folder_name}'
                seed = int(hashlib.sha256(seed_text.encode()).hexdigest()[:16], 16)
                rng = random.Random(seed)
                bit_count = int(payload * len(original_pixels))
                positions = rng.sample(range(len(original_pixels)), bit_count)
                message_bits = [rng.getrandbits(1) for _ in range(bit_count)]
                stego_pixels = original_pixels.copy()
                for position, bit in zip(positions, message_bits):
                    stego_pixels[position] = (stego_pixels[position] & ~1) | bit
                stego = Image.new('L', size)
                stego.putdata(stego_pixels)
                stego.save(output)
                with Image.open(output) as check:
                    saved_pixels = list(check.getdata())
                assert all((saved_pixels[p] & 1) == b
                           for p, b in zip(positions, message_bits))
                changed = sum(a != b for a, b in zip(original_pixels, saved_pixels))
                writer.writerow([source.name, payload, seed, bit_count, changed,
                                 source_hash, digest(output)])
            if number % 100 == 0:
                print(f'Processed {number}/500 originals')
    print('Finished: 500 originals, 500 images at 0.1 bpp, 500 at 0.4 bpp')
    print(f'Manifest: {manifest}')


def splits(desktop):
    """Assign source images to 300/100/100 groups with seed 123."""
    source_manifest = selected_dir(desktop) / 'source_manifest.csv'
    split_file = generated_dir(desktop) / 'splits.csv'
    with source_manifest.open(newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 500
    assert len({row['filename'] for row in rows}) == 500
    rows.sort(key=lambda row: row['filename'])
    random.Random(123).shuffle(rows)
    with split_file.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['source_filename', 'split', 'source_sha256'])
        for index, row in enumerate(rows):
            split = 'train' if index < 300 else 'validation' if index < 400 else 'test'
            writer.writerow([row['filename'], split, row['sha256']])
    print('Training sources: 300')
    print('Validation sources: 100')
    print('Test sources: 100')
    print(f'Split file: {split_file}')


def datasets(desktop):
    """Pair each source with its stego copy within the same split."""
    root = generated_dir(desktop)
    with (root / 'splits.csv').open(newline='') as f:
        sources = list(csv.DictReader(f))
    for payload in ('0p1', '0p4'):
        rows = []
        for source in sources:
            name, split = source['source_filename'], source['split']
            original = selected_dir(desktop) / name
            stego = root / payload / name
            assert original.is_file() and stego.is_file(), name
            rows.append([name, split, 'original', 0, str(original)])
            rows.append([name, split, 'stego', 1, str(stego)])
        output = root / f'dataset_{payload}.csv'
        with output.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['source_filename', 'split', 'type', 'label', 'path'])
            writer.writerows(rows)
        print(f'{payload}: {len(rows)} rows saved to {output.name}')


FEATURE_NAMES = ['pair_imbalance', 'pair_chisquare', 'horizontal_lsb_agreement',
                 'vertical_lsb_agreement', 'mean_horizontal_difference']


def extract_features(path):
    import numpy as np
    from PIL import Image
    with Image.open(path) as image:
        pixels = np.asarray(image, dtype=np.uint8)
    counts = np.bincount(pixels.ravel(), minlength=256)
    even, odd = counts[0::2], counts[1::2]
    pairs = even + odd
    pair_imbalance = np.abs(even - odd).sum() / pixels.size
    pair_chisquare = (((even - odd) ** 2 / np.maximum(pairs, 1)).sum()
                      / pixels.size)
    lsb = pixels & 1
    horizontal_agreement = np.mean(lsb[:, :-1] == lsb[:, 1:])
    vertical_agreement = np.mean(lsb[:-1, :] == lsb[1:, :])
    values = pixels.astype(np.int16)
    horizontal_difference = np.abs(values[:, 1:] - values[:, :-1]).mean()
    return [pair_imbalance, pair_chisquare, horizontal_agreement,
            vertical_agreement, horizontal_difference]


def features(desktop):
    """Extract five handcrafted image features for each payload dataset."""
    root = generated_dir(desktop)
    for payload in ('0p1', '0p4'):
        input_csv = root / f'dataset_{payload}.csv'
        output_csv = root / f'features_{payload}.csv'
        with input_csv.open(newline='') as f:
            images = list(csv.DictReader(f))
        with output_csv.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['source_filename', 'split', 'label'] + FEATURE_NAMES)
            for row in images:
                writer.writerow([row['source_filename'], row['split'], row['label']]
                                + extract_features(row['path']))
        print(f'{payload}: extracted features for {len(images)} images')
        print(f'Saved: {output_csv}')


def logistic(desktop):
    """Fit per-payload logistic models and evaluate the 0.5 threshold."""
    import joblib
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    root = generated_dir(desktop)
    for payload in ('0p1', '0p4'):
        with (root / f'features_{payload}.csv').open(newline='') as f:
            rows = list(csv.DictReader(f))
        feature_names = list(rows[0].keys())[3:]
        X = np.array([[float(row[name]) for name in feature_names] for row in rows])
        y = np.array([int(row['label']) for row in rows])
        group = np.array([row['split'] for row in rows])
        model = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=1000, random_state=42))
        model.fit(X[group == 'train'], y[group == 'train'])
        print(f'\nPayload: {payload}')
        for split in ('validation', 'test'):
            mask = group == split
            scores = model.predict_proba(X[mask])[:, 1]
            predictions = (scores >= 0.5).astype(int)
            tn, fp, fn, tp = confusion_matrix(y[mask], predictions,
                                               labels=[0, 1]).ravel()
            print(f'{split}: AUC={roc_auc_score(y[mask], scores):.3f}, '
                  f'F1={f1_score(y[mask], predictions):.3f}, '
                  f'false positives={fp}/100, missed stego={fn}/100')
        joblib.dump(model, root / f'logistic_{payload}.joblib')
    print('\nModels saved in BOSSbase_generated_500')


def rs_score(path):
    """Original shared simplified RS statistic, not full RS payload estimation."""
    import numpy as np
    from PIL import Image
    mask = np.array([0, 1, 1, 0], dtype=bool)[None, None, :]
    with Image.open(path) as image:
        pixels = np.asarray(image, dtype=np.int16)
    assert pixels.shape == (512, 512)
    groups = pixels.reshape(512, 128, 4)

    def smoothness(values):
        return np.abs(np.diff(values, axis=2)).sum(axis=2)

    before = smoothness(groups)
    positive = np.where(mask, groups ^ 1, groups)
    negative = np.where(mask, groups + np.where(groups % 2 == 0, -1, 1), groups)

    def difference(flipped):
        after = smoothness(flipped)
        return np.mean(after > before) - np.mean(after < before)

    return difference(negative) - difference(positive)


def rs(desktop):
    """Score images and set a threshold using validation originals only."""
    import numpy as np
    from sklearn.metrics import roc_auc_score
    root = generated_dir(desktop)
    with (root / 'splits.csv').open(newline='') as f:
        sources = list(csv.DictReader(f))
    original_scores = {source['source_filename']:
                       rs_score(selected_dir(desktop) / source['source_filename'])
                       for source in sources}
    output = root / 'rs_scores.csv'
    with output.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['source_filename', 'split', 'payload', 'label', 'rs_score'])
        for payload in ('0p1', '0p4'):
            results = {'validation': [], 'test': []}
            for source in sources:
                name, split = source['source_filename'], source['split']
                stego_score = rs_score(root / payload / name)
                cover_score = original_scores[name]
                writer.writerow([name, split, payload, 0, cover_score])
                writer.writerow([name, split, payload, 1, stego_score])
                if split in results:
                    results[split].extend([(0, cover_score), (1, stego_score)])
            validation_originals = [score for label, score in results['validation']
                                    if label == 0]
            threshold = np.quantile(validation_originals, 0.95)
            print(f'\n{payload} | validation threshold: {threshold:.5f}')
            for split in ('validation', 'test'):
                pairs = results[split]
                labels = np.array([label for label, score in pairs])
                scores = np.array([score for label, score in pairs])
                flagged = scores > threshold
                fp = np.sum((labels == 0) & flagged)
                detected = np.sum((labels == 1) & flagged)
                auc = roc_auc_score(labels, scores)
                print(f'{split}: AUC={auc:.3f}, false positives={fp}/100, '
                      f'stego detected={detected}/100')
    print(f'\nSaved scores: {output}')


def archive(desktop):
    """Package images and manifests in the layout expected by Colab."""
    root = generated_dir(desktop)
    files = (list(selected_dir(desktop).glob('*.pgm'))
             + [selected_dir(desktop) / 'source_manifest.csv']
             + list((root / '0p1').glob('*.pgm'))
             + list((root / '0p4').glob('*.pgm'))
             + [root / 'splits.csv', root / 'stego_manifest.csv'])
    assert len(files) == 1503, f'Expected 1503 files, found {len(files)}'
    output = desktop / 'BOSSbase_project_data.zip'
    with ZipFile(output, 'w', ZIP_DEFLATED, compresslevel=1) as z:
        for file in files:
            z.write(file, file.relative_to(desktop))
    print(f'Archive: {output}')
    print(f'Files: {len(files)}')
    print(f'Size (MB): {round(output.stat().st_size / 1_000_000, 1)}')


STAGES = dict(select=select, hashes=hashes, generate=generate, splits=splits,
              datasets=datasets, features=features, logistic=logistic,
              rs=rs, archive=archive)
STAGE_ORDER = tuple(STAGES)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--desktop', type=Path, required=True,
                        help='Parent of BOSSbase_1.01 and the generated folders')
    parser.add_argument('--stage', choices=(*STAGE_ORDER, 'all'), required=True,
                        help='Run one stage, or all stages in order on a fresh copy')
    args = parser.parse_args()
    desktop = args.desktop.expanduser().resolve()
    stages = STAGE_ORDER if args.stage == 'all' else (args.stage,)
    for stage in stages:
        print(f'\n[{stage}]')
        STAGES[stage](desktop)


if __name__ == '__main__':
    main()
