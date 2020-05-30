import os
from skimage.io import imread
from skimage.measure import compare_ssim
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt
import numpy as np
import sys

def l1_score(generated_images, reference_images):
    score_list = []
    for reference_image, generated_image in zip(reference_images, generated_images):
        score = np.abs(2 * (reference_image/255.0 - 0.5) - 2 * (generated_image/255.0 - 0.5)).mean()
        score_list.append(score)
    return np.mean(score_list)

def ssim_score(generated_images, reference_images):
    ssim_score_list = []
    for reference_image, generated_image in zip(reference_images, generated_images):
        ssim = structural_similarity(reference_image, generated_image, gaussian_weights=True, sigma=1.5,
                            use_sample_covariance=False, multichannel=True,
                            data_range=generated_image.max() - generated_image.min())
        ssim_score_list.append(ssim)
    return np.mean(ssim_score_list)

def ssim_score_v2(generated_images, reference_images):
    ssim_score_list = []
    for reference_image, generated_image in zip(reference_images, generated_images):
        ssim = structural_similarity(reference_image, generated_image, gaussian_weights=True, sigma=1.5,
                            use_sample_covariance=False, multichannel=True,
                            data_range=generated_image.max() - generated_image.min())
        ssim_score_list.append(ssim)
    return np.mean(ssim_score_list)

def load_generated_images(img_dir, epoch, ids):
    target_images = []
    generated_images = []

    for idv in ids:
        img = imread(os.path.join(img_dir, "epoch%s_%s.png" % (epoch, idv)))
        w = int(img.shape[1] / 5) #h, w ,c
        target_images.append(img[:, 2*w:3*w])
        generated_images.append(img[:, 4*w:5*w])

    return target_images, generated_images


def test_and_plot(img_dir, epochs, ids, fig_dir):

    tags = ['L1 loss', 'SSIM']
    scores = {
        tags[0]: [],
        tags[1]: []
    }
    for e in epochs:
        target_images, generated_images = load_generated_images(img_dir, e, ids)
        structured_score = ssim_score(generated_images, target_images)
        scores['SSIM'].append(structured_score)
        norm_score = l1_score(generated_images, target_images)
        scores['L1 loss'].append(norm_score)

    plt.figure(figsize=(20,10))
    for tag in tags:
        plt.plot(scores[tag], label=tag)
        plt.xlabel('epoch', fontsize=14)
        plt.ylabel('accuracy', fontsize=14)
    plt.legend(loc='best', fontsize=14)
    plt.grid(True)
    plt.title("Generator results accuracy", fontsize=20)
    plt.savefig(os.path.join(fig_dir, "generator_acc_metric.png"))
    plt.show()

if __name__ == "__main__":
    # img_dir = "Pose-Transfer/gan_model_e400/images"
    img_dir = sys.argv[1]
    # fig_dir = "Pose-Transfer/figures"
    fig_dir = sys.argv[2]
    epochs = ["%.3d" % idx for idx in np.arange(401)]
    ids = ids = ['0188', '0763', '0267', '0577', '0292', '0925', '0993', '0137', '0260', '0725', '1070', '1362', '0092', '0694', '0089', '0155']

    test_and_plot(img_dir, epochs, ids, fig_dir)






