from datasets import load_dataset

data_root = "/scratch/pf2m24/data/ADE20K/ADE20K/data"   # 就是你 screenshot 里的 data 目录

dataset_dict = load_dataset(
    "parquet",
    data_files={
        "train": f"{data_root}/train-*.parquet",
        "val":   f"{data_root}/validation-*.parquet",
    },
)

train_hfds = dataset_dict["train"]
val_hfds   = dataset_dict["val"]

print(train_hfds.column_names)
# 看一下是不是有 'image' 这一列
