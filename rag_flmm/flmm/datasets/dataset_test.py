import csv
import json
import random
from typing import Mapping

import numpy as np
import PIL
import torch
from torch.utils.data.dataloader import default_collate

GLD_image_path = "/PATH/TO/GLDv2"
iNat_image_path = "/PATH/TO/inaturalist"
infoseek_train_path = "/PATH/TO/InfoSeek/train"


def qformer_collate_fn(batch: list):
    """Discard None images in a batch when using torch DataLoader

    Args:
        batch (list): list of samples
    """
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


import os


def get_image(image_id, dataset_name, iNat_id2name=None):
    """Get the image file by image_id.
    Dataset details:
        iNaturalist:
            image ids are mapped by id2name dict to get the path of the image

        GLDv2/landmarks:
            image ids are indexed by its first 3 letters in the corresponding folder. e.g. image_id = "abcde" will be stored in "a/b/c/abcde.jpg"

        infoseek:
            image ids are either with suffix of .jpg or .JPEG

    Args:
        image_id : the image id
        dataset_name : the dataset name
        iNat_id2name : the dict to map image id to image name for iNaturalist dataset
    """
    if dataset_name == "inaturalist":
        file_name = iNat_id2name[image_id]
        image_path = os.path.join(iNat_image_path, file_name)
    elif dataset_name == "landmarks":
        image_path = os.path.join(
            GLD_image_path, image_id[0], image_id[1], image_id[2], image_id + ".jpg"
        )
    elif dataset_name == "infoseek":
        if os.path.exists(os.path.join(infoseek_train_path, image_id + ".jpg")):
            image_path = os.path.join(infoseek_train_path, image_id + ".jpg")
        elif os.path.exists(os.path.join(infoseek_train_path, image_id + ".JPEG")):
            image_path = os.path.join(infoseek_train_path, image_id + ".JPEG")
    else:
        raise NotImplementedError("dataset name not supported")
    return image_path


def reconstruct_wiki_article_dict(knowledge_entry):
    """Reconstruct the wiki article from the knowledge entry dict.

    Args:
        knowledge_entry : the knowledge entry dict
    """
    title = knowledge_entry["title"]
    article = "# Wiki Article: " + title + "\n"
    for it, section_title in enumerate(knowledge_entry["section_titles"]):
        article += (
            "\n## Section Title: "
            + section_title
            + "\n"
            + knowledge_entry["section_texts"][it]
        )

    return article


def reconstruct_wiki_sections_dict(knowledge_entry, section_index=-1):
    """Reconstruct the wiki sections from the knowledge entry dict.

    Args:
        knowledge_entry : the knowledge entry dict
    """
    title = knowledge_entry["title"]
    for it, section_title in enumerate(knowledge_entry["section_titles"]):
        if it == int(section_index):
            evidence_section = (
                "# Wiki Article: "
                + title
                + "\n"
                + "## Section Title: "
                + section_title
                + "\n"
                + knowledge_entry["section_texts"][it]
            )

    return evidence_section


class QFormerRerankerDataset(torch.utils.data.Dataset):
    """Dataset for training to retrieve semantically similar text.
    Used for training the QFormer model.
    """

    def __init__(
        self,
        knowledge_base_file,
        train_file,
        preprocess: callable,
        get_image_function=get_image,
        retriever=None,
        visual_attr_file=None,
        neg_num=4,
        inat_id2name=None,
    ):
        """Initialize the dataset.

        Args:
            knowledge_base_file (str): The path to the knowledge base file.
            train_file (str): The path to the train file.
            preprocess (callable): A callable function for preprocessing the data.
            get_image_function (callable, optional): A callable function for getting the image. Defaults to get_image.
            negative_db_file (str, optional): The path to the negative database file. Defaults to None.
            retriever (object, optional): An object for retrieving data. Defaults to None.
            visual_attr_file (str, optional): The path to the visual attribute file. Defaults to None.
            use_negative (bool, optional): Whether to use negative examples. Defaults to False.
            neg_num (int, optional): The number of negative examples to use. Defaults to 4.
            inat_id2name (str, optional): The path to the iNat ID to name mapping file. Defaults to None.
        """
        # load the knowledge base
        with open(knowledge_base_file, "r") as f:
            self.knowledge_base = json.load(f)
        self.kb_keys = list(self.knowledge_base.keys())
        self.train_list = []
        self.url_list = []
        with open(train_file, "r") as f:
            reader = csv.reader(f)
            self.header = next(reader)
            for row in reader:
                if (
                    row[self.header.index("question_type")] == "automatic"
                    or row[self.header.index("question_type")] == "templated"
                    or row[self.header.index("question_type")] == "multi_answer"
                    or row[self.header.index("question_type")] == "infoseek"
                ):
                    self.url_list.append(row[self.header.index("wikipedia_url")])
                    self.train_list.append(row)

        self.retriever = retriever
        self.preprocess = preprocess
        self.get_image = get_image_function
        self.max_length = 512

        if visual_attr_file is not None:
            with open(visual_attr_file, "r") as f:
                self.visual_attr = json.load(f)
        else:
            self.visual_attr = None

        self.neg_num = neg_num
        if inat_id2name is not None:
            with open(inat_id2name, "r") as f:
                self.iNat_id2name = json.load(f)
        else:
            self.iNat_id2name = None

    def __len__(self):
        return len(self.train_list)

    def get_url_list(self):
        return self.url_list

    def __getitem__(self, idx):
        print(f"getitem called with idx={idx}")
        print(f"type(self.train_list): {type(self.train_list)}")

        if isinstance(self.train_list, dict):
            print("Possible source of error: self.train_list is a dict, not a list.")
            print(f"Available keys: {list(self.train_list.keys())[:5]}")
        example = self.train_list[idx]
        question = example[self.header.index("question")]
        question_images = [
            self.get_image(
                image_id, example[self.header.index("dataset_name")], self.iNat_id2name
            )
            for image_id in example[self.header.index("dataset_image_ids")].split("|")
        ]
        question_image_path = question_images[0]
        question_image = self.preprocess(PIL.Image.open(question_image_path))

        positive_url = example[self.header.index("wikipedia_url")]
        positive_entry = self.knowledge_base[positive_url]
        evidence_section_id = example[self.header.index("evidence_section_id")]
        positive_section = reconstruct_wiki_sections_dict(
            positive_entry, evidence_section_id
        )

        return (
            question_image,
            question,
            positive_section,
            positive_url,
        )
