import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple
from concurrent.futures import ProcessPoolExecutor
import cv2
import imutils
import numpy as np


class YoloGen:
    def __init__(self, root_dir: str, start_class_id: int):
        self.images = tuple(img for img in Path(root_dir).glob('*.jpg'))
        self.angles = (0, 90, 180, 270)

        self.rotators = []

        for i, angle in enumerate(self.angles):
            rotator_dir = Path(root_dir) / str(angle)
            self.rotators.append(
                Rotator(rotator_dir, self.images, start_class_id + i, angle))


class Rotator:
    def __init__(self, rotator_dir: Path, images: Tuple[Path], class_id: int,
                 angle: int):
        self.class_id = class_id
        self.angle = angle
        self.images = images
        self.rotator_dir = rotator_dir

        Path.mkdir(rotator_dir, exist_ok=True)

    def processing(self) -> None:
        print(f'rotating image on {self.angle} degrees and generating yolofile')
        for image in self.images:
            rotated_img_abspath = image.parent / str(
                self.angle) / f'{image.stem}--{self.angle}{image.suffix}'
            rotated_img = self._get_rotated_img_ndarray(image, self.angle)
            self._save_rotated_image(rotated_img_abspath, rotated_img)
            self._calc_and_write_yolo_data(self.class_id, rotated_img_abspath,
                                           rotated_img)

    @staticmethod
    def _calc_and_write_yolo_data(class_id: int, image_abspath: Path,
                                  image: np.ndarray) -> None:
        img_h, img_w = image.shape[:2]
        # padding for yolo_bbox
        padding = 20

        # yolo data need relative to img shape
        x_center = img_w / 2 / img_w
        y_center = img_h / 2 / img_h

        bbox_h = round((img_h - padding) / img_h, 6)
        bbox_w = round((img_w - padding) / img_w, 6)

        yolo_data = f"{class_id} {x_center} {y_center} {bbox_w} {bbox_h}".encode()

        with open(image_abspath.with_suffix('.txt'), 'wb') as yolo_file:
            yolo_file.write(yolo_data)

    @staticmethod
    def _get_rotated_img_ndarray(image: Path, angle: int) -> np.ndarray:
        with open(image, 'rb') as img_b:
            # -1 cv2.IMREAD_UNCHANGED
            img = cv2.imdecode(
                np.asarray(bytearray(img_b.read()), dtype=np.uint8), -1)

            return img if angle == 0 else imutils.rotate_bound(img, angle)

    @staticmethod
    def _save_rotated_image(img_abspath: Path, img: np.ndarray) -> None:
        with open(img_abspath, 'wb') as result:
            buf = cv2.imencode('.jpg', img)[1].tobytes()
            result.write(buf)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('root_dir', type=str,
                        help='specify path of the processing images')
    parser.add_argument(
        'start_class_id', type=int,
        help='specify start_class_id for processing doctype according obj.names'
    )
    parser.add_argument('-c', dest='cores', type=int,
                        help='specify multiprocessing number of cores',
                        default=None)
    args = parser.parse_args()

    yologen = YoloGen(args.root_dir, args.start_class_id)

    start = time.time()
    # optional specify numbers of core
    with ProcessPoolExecutor(args.cores) as pool:
        for rotator in yologen.rotators:
            pool.submit(rotator.processing)

    print(f'elapsed time is {round(time.time() - start, 3)} sec')
