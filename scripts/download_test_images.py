import os
import urllib.request
import json
from PIL import Image

def download_test_suite():
    os.makedirs('test_images/real', exist_ok=True)
    os.makedirs('test_images/fake', exist_ok=True)

    hidf_samples = [
        'c01276_c01230.jpg', 'c01280_c01245.jpg', 'c01282_c01268.jpg', 'c01283_c01242.jpg',
        'c01284_c01252.jpg', 'c01285_c01249.jpg', 'c01287_c01260.jpg', 'c01291_c01313.jpg',
        'c01292_c01349.jpg', 'c01294_c01387.jpg', 'c01295_c01321.jpg', 'c01296_c01201.jpg',
        'c01297_c01228.jpg', 'c01299_c01283.jpg', 'c01300_c01236.jpg'
    ]

    manifest = {'real': [], 'fake': []}

    print('Downloading fake test images from HiDF...')
    for fname in hidf_samples:
        url = f'https://raw.githubusercontent.com/DSAIL-SKKU/HiDF/master/samples/images/{fname}'
        out_path = os.path.join('test_images/fake', f'test_fake_{fname}')
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=12) as resp, open(out_path, 'wb') as f:
                f.write(resp.read())
            im = Image.open(out_path)
            manifest['fake'].append({
                'filename': f'test_fake_{fname}',
                'path': out_path.replace('\\', '/'),
                'source': 'HiDF Benchmark (Human-Indistinguishable Deepfake)',
                'label': 'fake',
                'dimensions': f'{im.width}x{im.height}',
                'architecture': 'Neural Face-Swap / Synthesis'
            })
            print(f'  [OK] {fname} ({im.width}x{im.height})')
        except Exception as e:
            print(f'  [FAIL] {fname}: {e}')

    real_sources = [
        ('test_real_biden.jpg', 'https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/biden.jpg', 'Official Camera Portrait (High Res)'),
        ('test_real_obama.jpg', 'https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/obama.jpg', 'Official Presidential Camera Portrait'),
        ('test_real_alex.png', 'https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/alex-lacamoire.png', 'Genuine Portrait (Stage Light)'),
        ('test_real_lin.png', 'https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/lin-manuel-miranda.png', 'Natural Lighting Human Portrait'),
        ('test_real_einstein.jpg', 'https://upload.wikimedia.org/wikipedia/commons/d/d3/Albert_Einstein_Head.jpg', 'Historical Camera Optical Sensor'),
        ('test_real_curie.jpg', 'https://upload.wikimedia.org/wikipedia/commons/7/7e/Marie_Curie_c1920.jpg', 'Silver Halide Film Chemical Sensor'),
        ('test_real_turing.jpg', 'https://upload.wikimedia.org/wikipedia/commons/a/a1/Alan_Turing_Aged_16.jpg', 'Monochrome Film Portrait'),
        ('test_real_lovelace.png', 'https://upload.wikimedia.org/wikipedia/commons/0/0b/Ada_Byron_daguerreotype_by_Antoine_Claudet_1843_or_1850_-_cropped.png', 'Early Daguerreotype Sensor'),
        ('test_real_bohr.jpg', 'https://upload.wikimedia.org/wikipedia/commons/6/6d/Niels_Bohr.jpg', 'Genuine Studio Portrait'),
        ('test_real_feynman.jpg', 'https://upload.wikimedia.org/wikipedia/commons/1/1a/RichardFeynman-PVP.jpg', 'Genuine Public Archive Portrait')
    ]

    print('\nDownloading real test images...')
    for fname, url, desc in real_sources:
        out_path = os.path.join('test_images/real', fname)
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'SentinelResearch/1.0 (harshvshah2019@gmail.com)'})
            with urllib.request.urlopen(req, timeout=12) as resp, open(out_path, 'wb') as f:
                f.write(resp.read())
            im = Image.open(out_path)
            manifest['real'].append({
                'filename': fname,
                'path': out_path.replace('\\', '/'),
                'source': desc,
                'label': 'real',
                'dimensions': f'{im.width}x{im.height}',
                'architecture': 'Authentic Physical Camera / Sensor'
            })
            print(f'  [OK] {fname} ({im.width}x{im.height})')
        except Exception as e:
            print(f'  [FAIL] {fname}: {e}')

    with open('test_images/manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    total_real = len(manifest['real'])
    total_fake = len(manifest['fake'])
    print(f'\nDownload complete! Real test specimens: {total_real}, Fake test specimens: {total_fake}')

if __name__ == '__main__':
    download_test_suite()
