import pathlib
import logging

PROJECT_ROOT = pathlib.Path("/Users/xuxin/Documents/Dice/")

INPUT_DIR = PROJECT_ROOT / "0726_Raw"
OUTPUT_DIR = PROJECT_ROOT / "0726"

RANGES_INCLUSIVE = [
    (8346, 9999),
    (1, 775),
]


def get_image_path(dir: pathlib.Path, image_number: int) -> pathlib.Path:
    """
    Get the path to an image file based on the directory and image number.
    """
    return dir / f"IMG_{image_number:04d}.JPG"


def copy_image(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    if not output_path.parent.exists():
        raise FileNotFoundError(f"Output directory does not exist: {output_path.parent}")

    output_path.write_bytes(input_path.read_bytes())


def driver():
    dry_run = False
    # Check output directory is empty
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {OUTPUT_DIR}")

    output_index = 1

    for RANGE_INCLUSIVE in RANGES_INCLUSIVE:
        for input_index in range(RANGE_INCLUSIVE[0], RANGE_INCLUSIVE[1] + 1):
            input_path = get_image_path(INPUT_DIR, input_index)
            output_path = get_image_path(OUTPUT_DIR, output_index)

            try:
                if not dry_run:
                    copy_image(input_path, output_path)
                    logging.info(f"Copied {input_path} to {output_path}")
                else:
                    logging.info(f"Would copy {input_path} to {output_path}")
            except FileNotFoundError as e:
                logging.error(str(e))

            output_index += 1
            

if __name__ == "__main__":
    driver()