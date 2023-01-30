from embeding_search import EmbeddingSearch
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import json
import argparse
from types import SimpleNamespace
from tqdm import tqdm


def generate_labels(img_path, dest_dir, search_model, k_neigbor=10):
    neighbor_id_list = search_model.get_sim_img(img_path, k_neighbor=k_neigbor)
    query_img_stem = Path(img_path).stem
    img = Image.open(img_path).convert('RGB')
    fig, axes = plt.subplots(nrows=1, ncols=k_neigbor + 1, figsize=(k_neigbor * 2, 2 * 1))
    axes[0].imshow(img)
    axes[0].set_axis_off()
    # label = neighbor_id_list[0]
    for i in range(k_neigbor):
        neighbor_img = Image.open(neighbor_id_list[i]).convert('RGB')
        axes[i+1].imshow(neighbor_img)
        axes[i+1].set_axis_off()
    label_index_list = [neighbor_id.stem for neighbor_id in neighbor_id_list]
    label_index_str = "_".join(label_index_list)
    plt.savefig(f"{dest_dir}/{query_img_stem}_{label_index_str}.png")
    plt.close()


def run(args, embedding_search_model):
    dest_dir = "data/output_test"
    Path(dest_dir).mkdir(exist_ok=True, parents=True)
    img_path_list = list(Path(args.img_dir).glob("*.png"))
    print(len(img_path_list))
    for img_path in tqdm(img_path_list):
        generate_labels(str(img_path), str(dest_dir), embedding_search_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True, help="Path to config json file")
    parser.add_argument("--img_dir", type=str)

    args = parser.parse_args()

    with open(args.cfg, "r") as f:
        cfg = json.loads(f.read(), object_hook=lambda d: SimpleNamespace(**d))
    embedding_search_model = EmbeddingSearch(cfg)
    run(args, embedding_search_model)

