import tensorflow as tf
import numpy as np
import os
import cv2
from helper import gen_test_output, save_inference_samples

def load_model_checkpoint(sess, vgg_path):
    # Load the VGG model from the checkpoint
    model = tf.train.import_meta_graph(vgg_path + ".meta")
    model.restore(sess, vgg_path)
    print('Model loaded from checkpoint.')

    # Retrieve the tensors
    graph = tf.get_default_graph()
    input_image = graph.get_tensor_by_name('image_input:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    logits = graph.get_tensor_by_name('fcn_logits:0')

    return input_image, keep_prob, logits

def preprocess_image(image_path, image_shape):
    """
    Preprocess the input image
    """
    image = cv2.imread(image_path)
    image = cv2.resize(image, (image_shape[1], image_shape[0]))
    return image

def run_inference(sess, checkpoint_path, test_images_dir, output_dir, image_shape):
    # Load the model
    input_image, keep_prob, logits = load_model_checkpoint(sess, checkpoint_path)

    # Process test images and save results
    save_inference_samples(output_dir, test_images_dir, sess, image_shape, logits, keep_prob, input_image)

if __name__ == "__main__":
    # Paths and image shape
    vgg_path = './vgg/checkpoints/final_1.ckpt'
    test_images_dir = './data/test_images'
    output_dir = './data/output_images'
    image_shape = (3584, 3584)  # Adjust according to your model's expected input shape

    # Start a session and run inference
    with tf.Session() as sess:
        run_inference(sess, vgg_path, test_images_dir, output_dir, image_shape)