from tensorboard.backend.event_processing import event_accumulator
import sys
import pandas as pd
import os
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

size_guidance = {
        event_accumulator.COMPRESSED_HISTOGRAMS: 1,
        event_accumulator.IMAGES: 1,
        event_accumulator.AUDIO: 1,
        event_accumulator.SCALARS: 0,
        event_accumulator.HISTOGRAMS: 1,
    }

fig_dir = "figures"
os.makedirs(fig_dir, exist_ok=True)

def extract_and_plot_loss(log_file):
    ea = event_accumulator.EventAccumulator(log_file, size_guidance = size_guidance)
    ea.Reload()

    to_plot = [
        # {
        #     "name": "discriminator_loss_e100.png",
        #     "plots": [
        #         {
        #             "title": "Appearance discriminator loss",
        #             "tags": ['D_PP_GANloss/epoch/train', 'D_PP_GANloss/epoch/test']
        #         },
        #         {
        #             "title": "Shape discriminator loss",
        #             "tags": ['D_PB_GANloss/epoch/train', 'D_PB_GANloss/epoch/test']
        #         }
        #     ]
        # },
        {
            "name": "generator_loss_e100.png",
            "plots": [
                {
                    "title": "Generator combined L1 loss",
                    "tags": [
                        'G_L1loss/epoch/train',
                        'G_origin_L1/epoch/train',
                        'G_perceptual_L1/epoch/train',
                        'G_L1loss/epoch/test'
                        ]
                },
                {
                    "title": "Generator combined adversarial loss",
                    "tags": [
                        'G_GANloss/epoch/train',
                        'G_PP_GANloss/epoch/train',
                        'G_PB_GANloss/epoch/train',
                        'G_GANloss/epoch/test'
                        ]
                }
            ]
        },
        {
            "name": "gan_adversarial_loss_e100.png",
            "plots": [
                {
                    "title": "Generator and discriminators adversarial loss",
                    "tags": ['D_PP_GANloss/epoch/train', 'D_PB_GANloss/epoch/train', 'G_GANloss/epoch/train']
                }
            ]
        }
    ]

    for fig in to_plot:
        plt.figure(figsize=(20,10))
        plots = fig['plots']

        for plt_idx, plot in enumerate(plots):
            plt.subplot(len(plots), 1, plt_idx + 1)

            for tag in plot['tags']:
                scalar = ea.Scalars(tag)
                x_vals = [e.step for e in scalar]
                y_vals = [e.value for e in scalar]
                plt.plot(x_vals, y_vals, label = tag)
                plt.legend(loc='lower right', fontsize=14)

            plt.xlabel('epoch', fontsize=14)
            plt.ylabel('loss', fontsize=14)
            plt.title(plot['title'], fontsize = 20)
            plt.grid(True)

        plt.tight_layout()
        fig_path = os.path.join(fig_dir, fig['name'])
        plt.savefig(fig_path)
        plt.show()


log_file = sys.argv[1]
extract_and_plot_loss(log_file)

