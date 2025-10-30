import json
import os
import numpy as np
from PIL import Image
import chardet
from concurrent.futures import ThreadPoolExecutor
from skimage.draw import polygon


def create_mask_from_objects(objects, width, height):
    """
    Creates a segmentation mask from ISAT objects.
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    for obj in objects:
        segmentation = obj['segmentation']
        if not segmentation:
            continue

        poly = np.array(segmentation)
        if not np.array_equal(poly[0], poly[-1]):
            poly = np.vstack((poly, poly[0]))

        rr, cc = polygon(poly[:, 1], poly[:, 0], mask.shape)
        mask[rr, cc] = 255

    return mask


def process_file(file_path, save_path, json_path):
    try:
        with open(file_path, 'rb') as f:
            content = f.read()

        result = chardet.detect(content)
        try:
            data = json.loads(content.decode(result['encoding']))
        except:
            for encoding in ['utf-8', 'gbk', 'iso-8859-1', 'latin-1']:
                try:
                    data = json.loads(content.decode(encoding))
                    break
                except:
                    continue
            else:
                print(f"File {os.path.basename(file_path)} could not be decoded at all")
                return False

        info = data.get('info', {})
        if not info:
            print(f"File {os.path.basename(file_path)} is missing the 'info' field")
            return False

        width = info.get('width', 0)
        height = info.get('height', 0)
        if width == 0 or height == 0:
            print(f"File {os.path.basename(file_path)} is missing valid width/height")
            return False

        objects = data.get('objects', [])
        mask = create_mask_from_objects(objects, width, height)

        json_name = os.path.basename(file_path).split('.')[0]
        mask_img = Image.fromarray(mask).convert('L')
        mask_img.save(os.path.join(save_path, f"{json_name}.png"))
        return True

    except Exception as e:
        print(f"Error processing file {os.path.basename(file_path)}: {str(e)}")
        return False


def json2mask_multi(json_path, save_path, max_workers=4):
    """
    Converts JSON annotation files in a directory to binary masks using multithreading.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    files = [os.path.join(json_path, f) for f in os.listdir(json_path)
             if f.endswith('.json')]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_file, file, save_path, json_path) for file in files]
        for future in futures:
            future.result()


json_path = r"json_datafiles"
save_path = r"mask_datafiles"
json2mask_multi(json_path, save_path, max_workers=4)
