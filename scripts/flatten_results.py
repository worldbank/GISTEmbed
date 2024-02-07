from pathlib import Path
import shutil
import fire
from tqdm.auto import tqdm


def move_results(path):

    path = Path(path)
    assert path.is_dir()

    dirname = path.name
    flat = path.parent.parent / "flat" / dirname

    flat.mkdir(parents=True, exist_ok=True)

    for fp in tqdm(path.glob("*/*.json")):
        name = fp.name

        if name.endswith("scores.json"):
            continue

        dst = flat / name

        shutil.copy(fp, dst)



if __name__ == "__main__":
    fire.Fire(move_results)
