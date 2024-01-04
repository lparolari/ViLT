import json
import pandas as pd
import pyarrow as pa
import random
import os

from tqdm import tqdm
from glob import glob
from collections import defaultdict


def path2rest(iid, iid2path, iid2captions, iid2split, iid2boxes, root):
    name = iid2path[iid]
    path = f"{root}/{name}"
 
    with open(path, "rb") as fp:
        binary = fp.read()

    captions = iid2captions[iid]
    split = iid2split[iid]
    box = iid2boxes[iid]

    return [binary, captions, name, box, split]


def make_arrow(root, dataset_root):
    for split in ["train", "val", "test"]:
        

        with open(f"{root}/mdetr/instances_{split}.json", "r") as fp:
            data = json.load(fp)

        images = data["images"]
        annotations = data["annotations"]

        imgid2idx = dict()
        out = []
        binaries = {}

        for i, img in enumerate(images):
            imgid2idx[img["id"]] = i

        for ann in tqdm(annotations):
            img_id = ann["image_id"]
            img_ann = images[imgid2idx[img_id]]

            caption = img_ann["caption"]

            query_pos = ann["tokens_positive"][0]
            query_start, query_end = query_pos

            query = caption[query_start:query_end]
            box = ann["bbox"]
            x, y, w, h = box
            img_w, img_h = float(img_ann["width"]), float(img_ann["height"])
            box = [x / img_w, y / img_h, w / img_w, h / img_h]  # TODO: we should transform boxes later

            # path = f"{root}/flickr30k-images/{img_ann['file_name']}"
            # with open(path, "rb") as fp:
            #     binary = fp.read()
            #     if img_id not in binaries:
            #         binaries[img_id] = binary

            out.append({
                "image": img_ann['file_name'],  # TODO: tmp
                "caption": query,
                "image_id": img_id,
                "box": box,
                "split": split,
            })

        batches = [[b["image"], b["caption"], b["image_id"], b["box"], b["split"]] for b in out]
        random.shuffle(batches)

        schema = pa.schema([
            pa.field("image", pa.string()),
            pa.field("caption", pa.list_(pa.string())),
            pa.field("image_id", pa.int64()),
            pa.field("box", pa.list_(pa.float32())),
            pa.field("split", pa.string()),
        ])

        os.makedirs(dataset_root, exist_ok=True)
        sink = pa.OSFile(
            f"{dataset_root}/f30k_caption_karpathy_{split}.arrow", "wb"
        )
        writer = pa.RecordBatchFileWriter(sink, schema)
        
        for o in tqdm(out):
            batch = pa.RecordBatch.from_arrays(
                [pa.array([o["image"]]), pa.array([[o["caption"]]]), pa.array([o["image_id"]]), pa.array([o["box"]]), pa.array([o["split"]])],
                schema=schema
            )
            writer.write_batch(batch)
        
        writer.close()
        sink.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, required=True)
    args = parser.parse_args()

    make_arrow(args.root, args.dataset_root)