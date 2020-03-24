#!/usr/bin/env python3

import sys, os
from argparse import ArgumentParser, SUPPRESS
import cv2
import numpy as np
import logging as log
from time import time
from openvino.inference_engine import IENetwork, IECore

# args_model = "mobilenet_v2_12_fp16.xml"
args_model = "mobilenet_v2.xml"
args_input = "20180714014405.jpg"
args_device = "MYRIAD"

class NCSModel(object):
    def __init__(self, model_name):
        model_xml = model_name
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        ie = IECore()
        self.net = IENetwork(model=model_xml, weights=model_bin)

        assert len(self.net.inputs.keys()) == 1, "Sample supports only single input topologies"
        assert len(self.net.outputs) == 1, "Sample support only single output topologies"

        self.input_blob = next(iter(self.net.inputs))
        self.out_blob = next(iter(self.net.outputs))
        self.net.batch_size = 1

        self.exec_net = ie.load_network(network=self.net, device_name=args_device)

    def __call__(self, image):
        n, c, h, w = self.net.inputs[self.input_blob].shape
        images = np.ndarray(shape=(n, c, h, w))
        # for i in range(n):
            # image = cv2.imread(args_input)
        if image.shape[:-1] != (h, w):
            log.warning("Image is resized from {} to {}".format(image.shape[:-1], (h, w)))
            image = cv2.resize(image, (w, h))

        image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        images = np.asarray([image])
        log.info("Batch size is {}".format(n))

        log.info("Loading model to the plugin")

        log.info("Starting inference in synchronous mode")
        res = self.exec_net.infer(inputs={self.input_blob: images})

        # Processing output blob
        log.info("Processing output blob")
        res = res[self.out_blob]
        log.info("Top {} results: ".format(10))

        labels_map = None

        classid_str = "classid"
        probability_str = "probability"
        for i, probs in enumerate(res):
            probs = np.squeeze(probs)
            top_ind = np.argsort(probs)[-10:][::-1]
            # print(probs[np.argsort(probs)][::-1])
            # print(len(probs))

            # print("Image {}\n".format(args_input))
            # print(classid_str, probability_str)
            # print("{} {}".format('-' * len(classid_str), '-' * len(probability_str)))
            # for id in top_ind:
            #     det_label = labels_map[id] if labels_map else "{}".format(id)
            #     label_length = len(det_label)
            #     space_num_before = (len(classid_str) - label_length) // 2
            #     space_num_after = len(classid_str) - (space_num_before + label_length) + 2
            #     space_num_before_prob = (len(probability_str) - len(str(probs[id]))) // 2
            #     print("{}{}{}{}{:.7f}".format(' ' * space_num_before, det_label,
            #                                   ' ' * space_num_after, ' ' * space_num_before_prob,
            #                                   probs[id]))
            # print("\n")

        print(probs.shape)
        return probs

if __name__ == '__main__':
    model = NCSModel(args_model)
    image = cv2.imread(args_input)
    prob = model(image)
    print(prob)
    print(len(prob))
    sys.exit(0)
