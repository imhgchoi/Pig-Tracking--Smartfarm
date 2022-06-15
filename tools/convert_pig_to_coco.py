import os
import numpy as np
import json
import tqdm
from PIL import Image

DATA_PATH = 'datasets/pig/'
OUT_PATH = DATA_PATH + 'processed/annotations/'
SPLITS = ['train']
DEBUG = False

def load_func(fpath):
    print('fpath', fpath)
    assert os.path.exists(fpath)
    with open(fpath,'r') as fid:
        lines = fid.readlines()
    records =[json.loads(line.strip('\n')) for line in lines]
    return records

if __name__ == '__main__':
    if not os.path.exists(OUT_PATH):
        os.mkdir(OUT_PATH)
    for split in SPLITS:
        data_path = DATA_PATH + 'raw/'
        out_path = OUT_PATH + '{}.json'.format(split)
        out = {'images': [], 'annotations': [], 'categories': [{'id': 1, 'name': 'pig'}]}

        seqs = [f for f in os.listdir(data_path) if '.jpg' in f]
        seqs = [f for f in seqs if 'night' not in f]
        seqs.sort()

        image_cnt = 0
        ann_cnt = 0
        prev_vcnt, video_cnt = -1, 0
        pcam, pyr, pmo, pda, phr, pmn, pidx = 'none', -1, -1, -1, -1, -1, -1
        for i, fname in enumerate(tqdm.tqdm(seqs)) :
            image_cnt += 1
            file_path = data_path + fname
            im = Image.open(file_path)
            # im.save(DATA_PATH + 'processed/' + fname)

            cam, yr, mo, da, hr, mn, idx = fname.split('-')
            fid = 1 if (cam!=pcam or yr!=pyr or mo!=pmo or da!=pda or hr!=phr or mn!=pmn) else fid+1
            video_cnt = video_cnt+1 if (cam!=pcam or yr!=pyr or mo!=pmo or da!=pda or hr!=phr or mn!=pmn) else video_cnt
            if prev_vcnt != video_cnt : 
                prev_image_id = -1
            else :
                prev_image_id = image_cnt - 1
            if len(seqs) == i+1 :
                next_image_id = -1
            else :
                ncam, nyr, nmo, nda, nhr, nmn, nidx = seqs[i+1].split('-')
                next_image_id = -1 if (cam!=ncam or yr!=nyr or mo!=nmo or da!=nda or hr!=nhr or mn!=nmn) else image_cnt + 1
                
            image_info = {'file_name': '{}'.format(fname), 
                          'id': image_cnt,
                          'frame_id' : fid,
                          'prev_image_id': prev_image_id,
                          'next_image_id': next_image_id,
                          'video_id' : video_cnt,
                          'height': im.size[1], 
                          'width': im.size[0]}
            out['images'].append(image_info)
            pcam, pyr, pmo, pda, phr, pmn, pidx = cam, yr, mo, da, hr, mn, idx
            prev_vcnt = video_cnt

            bbox = np.loadtxt(data_path + fname[:-4]+'.txt', dtype=str, delimiter=',')
            for box in bbox :
                ann_cnt += 1
                _, cx, cy, w, h = map(float, box.split(' '))
                ann =  {'id': ann_cnt,
                        'category_id': 1,
                        'image_id': image_cnt,
                        'track_id': -1,
                        'bbox': [int(im.size[0]*(cx - w/2)), int(im.size[1]*(cy - h/2)), int(im.size[0]*w), int(im.size[1]*h)],
                        'area': im.size[0] * w * im.size[1] * h,
                        'iscrowd': 0}
                out['annotations'].append(ann)
        print('loaded {} for {} images and {} samples'.format(split, len(out['images']), len(out['annotations'])))
        json.dump(out, open(out_path, 'w'))