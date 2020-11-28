import numpy as np
import logging
import pathlib
import cv2
import os


class COCODataset:

    def __init__(self, root, transform=None, target_transform=None, is_test=False,  label_file=None):
        """Dataset for VOC data.
        Args:
            root: the root of the COCO2014 or COCO14 dataset, the directory contains the following:
                -images
                    -train
                    -val
                -labels
                    -train 
                    -val
                all_images_train.txt -- training set
                all_images_val.txt -- validation set
                coco.names 
        """
        self.root = pathlib.Path(root)
        self.transform = transform
        self.target_transform = target_transform
        if is_test:
            image_sets_file = self.root / "all_images_val.txt"
            self.image_path = self.root / "images" / "val2017"
            self.label_path = self.root / "labels" / "val2017"
        else:
            image_sets_file = self.root / "all_images_train.txt"
            self.image_path = self.root / "images" / "train2017"
            self.label_path = self.root / "labels" / "train2017"

        self.ids = COCODataset._read_image_ids(image_sets_file)

        # if the labels file exists, read in the class names
        label_file_name = self.root / "coco.names"

        if os.path.isfile(label_file_name):
            classes = []
            with open(label_file_name, 'r') as infile:
                for line in infile:
                    classes.append(line.rstrip())

                classes.insert(0, 'BACKGROUND')
                self.class_names = tuple(classes)
                logging.info("COCO Labels read from file: " + str(self.class_names))
        else:
            logging.info("No labels file, using default COCO classes.")
            self.class_names = ("BACKGROUND", "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
                                "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush")

        self.class_dict = {class_name: i for i,
                           class_name in enumerate(self.class_names)}

    def _get_image_width_height(self, image_id):
        image = self._read_image(image_id)
        width = image.shape[1]
        height = image.shape[0]
        return width, height

    def __getitem__(self, index):
        image_id = self.ids[index]
        image = self._read_image(image_id)
        boxes, labels, is_difficult = self._get_annotation(image_id)
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        return image, boxes, labels

    def get_image(self, index):
        image_id = self.ids[index]
        image = self._read_image(image_id)
        if self.transform:
            image, _ = self.transform(image)
        return image

    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _read_image_ids(image_sets_file):
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                ids.append(line.rstrip().split('/')[-1].split('.')[0])
        return ids

    def _get_annotation(self, image_id):
        width, height = self._get_image_width_height(image_id)
        annotation_file = self.label_path / f"{image_id}.txt"
        boxes = []
        labels = []

        with open(annotation_file) as f:
            for line in f:
                # print(line)
                f_list = [float(i) for i in line.split(" ")]
                class_id = f_list[0] +1 
                
                x1_center = f_list[1] * width
                y1_center = f_list[2] * height
                w = f_list[3] * width 
                h = f_list[4] * height 

                x1 = x1_center - w/2
                x2 = x1_center + w/2
                y1 = y1_center - h/2
                y2 = y1_center + h/2
                
                boxes.append([x1, y1, x2, y2])
                labels.append(class_id)

                # print(boxes, labels)

        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64), 
                np.zeros(len(labels)))

    def _read_image(self, image_id):
        image_file = self.image_path / f"{image_id}.jpg"
        image = cv2.imread(str(image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
