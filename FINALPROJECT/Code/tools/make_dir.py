from pathlib import Path


def make_dir(dirname, f_ype):
    if f_ype == "train":
        output_dir = Path('./{}/train'.format(dirname))
        output_dir.mkdir(parents=True, exist_ok=True)

    elif f_ype == "test":
        output_dir = Path('./{}/test'.format(dirname))
        output_dir.mkdir(parents=True, exist_ok=True)

    elif f_ype == "dev":
        output_dir = Path('./{}/dev'.format(dirname))
        output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir
