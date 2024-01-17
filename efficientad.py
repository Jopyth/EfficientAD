#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import tifffile
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import itertools
import os
import random
from tqdm import tqdm
from common import get_autoencoder, get_pdn_small, get_pdn_medium, \
    ImageFolderWithoutTarget, ImageFolderWithPath, SingleFolderWithoutTarget, InfiniteDataloader, calculate_f1_max
from sklearn.metrics import roc_auc_score
from azureml.core import Workspace

import matplotlib.pyplot as plt
import csv
from PIL import Image
import mlflow
from mlflow import log_artifacts, log_metric, log_param, log_figure

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='mvtec_ad',
                        choices=['mvtec_ad', 'mvtec_loco'])
    parser.add_argument('-s', '--subdataset', default='bottle',
                        help='One of 15 sub-datasets of Mvtec AD or 5' +
                             'sub-datasets of Mvtec LOCO')
    parser.add_argument('-o', '--output_dir', default='output/1')
    parser.add_argument('-m', '--model_size', default='small',
                        choices=['small', 'medium'])
    parser.add_argument('-w', '--weights', default='models/teacher_small.pth')
    parser.add_argument('-i', '--imagenet_train_path',
                        default='none',
                        help='Set to "none" to disable ImageNet' +
                             'pretraining penalty. Or see README.md to' +
                             'download ImageNet and set to ImageNet path')
    parser.add_argument('-a', '--mvtec_ad_path',
                        default='./mvtec_anomaly_detection',
                        help='Downloaded Mvtec AD dataset')
    parser.add_argument('-b', '--mvtec_loco_path',
                        default='./mvtec_loco_anomaly_detection',
                        help='Downloaded Mvtec LOCO dataset')
    parser.add_argument('-t', '--train_steps', type=int, default=7000)

    parser.add_argument('-sf', '--stage_inference', action='store_true')
    parser.add_argument('-aug', '--img_aug', action='store_true')
    return parser.parse_args()

# constants
seed = 42
on_gpu = torch.cuda.is_available()
out_channels = 384
image_size = 256

# data loading
default_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
transform_ae = transforms.RandomChoice([
    transforms.ColorJitter(brightness=0.2),
    transforms.ColorJitter(contrast=0.2),
    transforms.ColorJitter(saturation=0.2)
])


# data augmentation transform
aug_transformations_mvtec_ad = [
    transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.95, 1.05)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.GaussianBlur(kernel_size=3)
]
aug_transformations_all = [
    transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.95, 1.05)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.GaussianBlur(kernel_size=3),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.1, contrast=0.1), transforms.RandomAffine(degrees=0, shear=0.1, scale=(1.0, 1.1))], p=0.5),
    transforms.RandomChoice([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()])
]


def train_transform(image):
    return default_transform(image), default_transform(transform_ae(image))

def main():
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    config = get_argparse()

    if config.dataset == 'mvtec_ad':
        dataset_path = config.mvtec_ad_path
    elif config.dataset == 'mvtec_loco':
        dataset_path = config.mvtec_loco_path
    else:
        raise Exception('Unknown config.dataset')

    pretrain_penalty = True
    if config.imagenet_train_path == 'none':
        pretrain_penalty = False


    # create output dir
    train_output_dir = os.path.join(config.output_dir, 'trainings',
                                    config.dataset, config.subdataset)
    test_output_dir = os.path.join(config.output_dir, 'anomaly_maps',
                                   config.dataset, config.subdataset, 'test')
    if not config.stage_inference:
        os.makedirs(train_output_dir, exist_ok=True)
        os.makedirs(test_output_dir, exist_ok=True)

    # data augmentation
    img_aug_path_to_check = os.path.join(dataset_path, config.subdataset, 'train', 'good_aug')
    if config.img_aug:
        if not os.path.exists(img_aug_path_to_check): 
            os.makedirs(img_aug_path_to_check)
            if config.dataset == 'mvtec_ad':
                for i, trans in enumerate(aug_transformations_mvtec_ad):
                    for input_image in os.listdir(os.path.join(dataset_path, config.subdataset, 'train', 'good')):
                        augmented_image = trans(Image.open(os.path.join(dataset_path, config.subdataset, 'train', 'good', input_image)))
                        augmented_image.save(os.path.join(img_aug_path_to_check, str(i) + '_' + input_image))
            else:
                for i, trans in enumerate(aug_transformations_all):
                    for input_image in os.listdir(os.path.join(dataset_path, config.subdataset, 'train', 'good')):
                        augmented_image = trans(Image.open(os.path.join(dataset_path, config.subdataset, 'train', 'good', input_image)))
                        augmented_image.save(os.path.join(img_aug_path_to_check, str(i) + '_' + input_image))
        else:
            print("the augmented images are already generated and saved in %s"%img_aug_path_to_check)

    # load data, with or without augmented data
    if config.img_aug:
        full_train_set = ImageFolderWithoutTarget(
            os.path.join(dataset_path, config.subdataset, 'train'),
            transform=transforms.Lambda(train_transform))
        print(f"loaded {len(full_train_set)} images (with augmentation).")
    else:
        full_train_set = SingleFolderWithoutTarget(
            os.path.join(dataset_path, config.subdataset, 'train'), 
            'good',
            transform=transforms.Lambda(train_transform))
        print(f"loaded {len(full_train_set)} images (without augmentation).")

    if config.dataset == 'mvtec_ad':
        # mvtec dataset paper recommend 10% validation set
        train_size = int(0.9 * len(full_train_set))
        validation_size = len(full_train_set) - train_size
        rng = torch.Generator().manual_seed(seed)
        train_set, validation_set = torch.utils.data.random_split(full_train_set,
                                                           [train_size,
                                                            validation_size],
                                                           rng)
    elif config.dataset == 'mvtec_loco':
        train_set = full_train_set
        validation_set = ImageFolderWithoutTarget(
            os.path.join(dataset_path, config.subdataset, 'validation'),
            transform=transforms.Lambda(train_transform))
    else:
        raise Exception('Unknown config.dataset')

    test_set = ImageFolderWithPath(
        os.path.join(dataset_path, config.subdataset, 'test'))


    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    train_loader_infinite = InfiniteDataloader(train_loader)
    validation_loader = DataLoader(validation_set, batch_size=1)

    if pretrain_penalty:
        # load pretraining data for penalty
        penalty_transform = transforms.Compose([
            transforms.Resize((2 * image_size, 2 * image_size)),
            transforms.RandomGrayscale(0.3),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,
                                                                  0.225])
        ])
        penalty_set = ImageFolderWithoutTarget(config.imagenet_train_path,
                                               transform=penalty_transform)
        penalty_loader = DataLoader(penalty_set, batch_size=1, shuffle=True,
                                    num_workers=4, pin_memory=True)
        penalty_loader_infinite = InfiniteDataloader(penalty_loader)
    else:
        penalty_loader_infinite = itertools.repeat(None)

    # create models
    if config.model_size == 'small':
        teacher = get_pdn_small(out_channels)
        student = get_pdn_small(2 * out_channels)
    elif config.model_size == 'medium':
        teacher = get_pdn_medium(out_channels)
        student = get_pdn_medium(2 * out_channels)
    else:
        raise Exception()
    state_dict = torch.load(config.weights, map_location='cpu')
    teacher.load_state_dict(state_dict)
    autoencoder = get_autoencoder(out_channels)

    if not config.stage_inference:
        teacher.eval()
        student.train()
        autoencoder.train()
    else:
        # only want to evaluate image classification after training
        teacher = torch.load(os.path.join(train_output_dir, 'teacher_final.pth'))
        student = torch.load(os.path.join(train_output_dir, 'student_final.pth'))
        autoencoder = torch.load(os.path.join(train_output_dir, 'autoencoder_final.pth'))
        teacher.eval()
        student.eval()
        autoencoder.eval()

    if on_gpu:
        teacher.cuda()
        student.cuda()
        autoencoder.cuda()

    teacher_mean, teacher_std = teacher_normalization(teacher, train_loader)

    if not config.stage_inference:
        optimizer = torch.optim.Adam(itertools.chain(student.parameters(),
                                                     autoencoder.parameters()),
                                     lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(0.95 * config.train_steps), gamma=0.1)
        tqdm_obj = tqdm(range(config.train_steps))

        workspace = Workspace.from_config()
        mlflow_tracking_uri = workspace.get_mlflow_tracking_uri()

        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment("EfficientAD")
        mlflow.start_run()
        print("Tracking URI:", mlflow_tracking_uri)

        params_to_log = [
            "dataset",
            "subdataset",
            "output_dir",
            "model_size",
            "imagenet_train_path",
            "train_steps",
            "stage_inference",
            "img_aug",
        ]

        param_log_dict = {key: vars(config)[key] for key in params_to_log}
        print(param_log_dict)
        mlflow.log_params(param_log_dict)

        train_loss_avg = 0.
        for iteration, (image_st, image_ae), image_penalty in zip(
                tqdm_obj, train_loader_infinite, penalty_loader_infinite):
            if on_gpu:
                image_st = image_st.cuda()
                image_ae = image_ae.cuda()
                if image_penalty is not None:
                    image_penalty = image_penalty.cuda()
            with torch.no_grad():
                teacher_output_st = teacher(image_st)
                teacher_output_st = (teacher_output_st - teacher_mean) / teacher_std
            student_output_st = student(image_st)[:, :out_channels]
            distance_st = (teacher_output_st - student_output_st) ** 2
            d_hard = torch.quantile(distance_st, q=0.999)
            loss_hard = torch.mean(distance_st[distance_st >= d_hard])

            if image_penalty is not None:
                student_output_penalty = student(image_penalty)[:, :out_channels]
                loss_penalty = torch.mean(student_output_penalty**2)
                loss_st = loss_hard + loss_penalty
            else:
                loss_st = loss_hard

            ae_output = autoencoder(image_ae)
            with torch.no_grad():
                teacher_output_ae = teacher(image_ae)
                teacher_output_ae = (teacher_output_ae - teacher_mean) / teacher_std
            student_output_ae = student(image_ae)[:, out_channels:]
            distance_ae = (teacher_output_ae - ae_output)**2
            distance_stae = (ae_output - student_output_ae)**2
            loss_ae = torch.mean(distance_ae)
            loss_stae = torch.mean(distance_stae)
            loss_total = loss_st + loss_ae + loss_stae

            train_loss_avg += loss_total.item()
            log_metric("Average training loss", train_loss_avg/(iteration+1), iteration+1)

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            scheduler.step()

            if iteration % 10 == 0:
                tqdm_obj.set_description(
                    "Current loss: {:.4f}  ".format(loss_total.item()))

            if iteration % 1000 == 0:
                torch.save(teacher, os.path.join(train_output_dir,
                                                 'teacher_tmp.pth'))
                torch.save(student, os.path.join(train_output_dir,
                                                 'student_tmp.pth'))
                torch.save(autoencoder, os.path.join(train_output_dir,
                                                     'autoencoder_tmp.pth'))

            if iteration % 1000 == 0 and iteration > 0:
                # run intermediate evaluation
                teacher.eval()
                student.eval()
                autoencoder.eval()

                q_st_start, q_st_end, q_ae_start, q_ae_end = map_normalization(
                    validation_loader=validation_loader, teacher=teacher,
                    student=student, autoencoder=autoencoder,
                    teacher_mean=teacher_mean, teacher_std=teacher_std,
                    desc='Intermediate map normalization')
                auc, f1 = test(
                    test_set=test_set, teacher=teacher, student=student,
                    autoencoder=autoencoder, teacher_mean=teacher_mean,
                    teacher_std=teacher_std, q_st_start=q_st_start,
                    q_st_end=q_st_end, q_ae_start=q_ae_start, q_ae_end=q_ae_end,
                    test_output_dir=None, desc='Intermediate inference')
                print('Intermediate image auc: {:.4f}, image F1: {:.4f}'.format(auc, f1))

                log_metric("Test image AUC", auc, iteration+1)
                log_metric("Test image F1", f1, iteration+1)

                # teacher frozen
                teacher.eval()
                student.train()
                autoencoder.train()

        teacher.eval()
        student.eval()
        autoencoder.eval()

        torch.save(teacher, os.path.join(train_output_dir, 'teacher_final.pth'))
        torch.save(student, os.path.join(train_output_dir, 'student_final.pth'))
        torch.save(autoencoder, os.path.join(train_output_dir, 'autoencoder_final.pth'))

        q_st_start, q_st_end, q_ae_start, q_ae_end = map_normalization(
            validation_loader=validation_loader, teacher=teacher, student=student,
            autoencoder=autoencoder, teacher_mean=teacher_mean,
            teacher_std=teacher_std, desc='Final map normalization')
        auc, f1 = test(
            test_set=test_set, teacher=teacher, student=student,
            autoencoder=autoencoder, teacher_mean=teacher_mean,
            teacher_std=teacher_std, q_st_start=q_st_start, q_st_end=q_st_end,
            q_ae_start=q_ae_start, q_ae_end=q_ae_end,
            test_output_dir=test_output_dir, desc='Final inference', config=config)

        log_metric("Test image AUC", auc, iteration)
        log_metric("Test image F1", f1, iteration)
        mlflow.log_artifacts(train_output_dir, "train")
        mlflow.log_artifacts(test_output_dir, "test")

        print('Final evaluation on test set, image classification auc: {:.4f}, F1: {:.4f}'.format(auc, f1))

        mlflow.end_run()

    else:
        q_st_start, q_st_end, q_ae_start, q_ae_end = map_normalization(
            validation_loader=validation_loader, teacher=teacher, student=student,
            autoencoder=autoencoder, teacher_mean=teacher_mean,
            teacher_std=teacher_std, desc='After training map normalization')
        auc, f1 = test(
            test_set=test_set, teacher=teacher, student=student,
            autoencoder=autoencoder, teacher_mean=teacher_mean,
            teacher_std=teacher_std, q_st_start=q_st_start, q_st_end=q_st_end,
            q_ae_start=q_ae_start, q_ae_end=q_ae_end,
            test_output_dir=test_output_dir, desc='Only for inference', config=config)
        print('Evaluation on test set, image classification auc: {:.4f}, F1: {:.4f}'.format(auc, f1))


def test(test_set, teacher, student, autoencoder, teacher_mean, teacher_std,
         q_st_start, q_st_end, q_ae_start, q_ae_end, test_output_dir=None,
         desc='Running inference', config=None):
    y_true = []
    y_score = []
    prediction_infos = []
    prediction_infos.append(['Defect type', 'Image Nr.', 'Groud truth', 'Prediction', 'Ground truth label', 'Anomaly score'])
    defect_types = []
    image_ids= []
    for image, target, path in tqdm(test_set, desc=desc):
        orig_width = image.width
        orig_height = image.height
        image = default_transform(image)
        image = image[None]
        if on_gpu:
            image = image.cuda()

        map_combined, map_st, map_ae = predict(
            image=image, teacher=teacher, student=student,
            autoencoder=autoencoder, teacher_mean=teacher_mean,
            teacher_std=teacher_std, q_st_start=q_st_start, q_st_end=q_st_end,
            q_ae_start=q_ae_start, q_ae_end=q_ae_end)
        map_combined = torch.nn.functional.pad(map_combined, (4, 4, 4, 4))
        map_combined = torch.nn.functional.interpolate(
            map_combined, (orig_height, orig_width), mode='bilinear')
        map_combined = map_combined[0, 0].cpu().numpy()

        defect_class = os.path.basename(os.path.dirname(path))
        defect_types.append(defect_class)
        image_ids.append(os.path.split(path)[1].split('.')[0])
        if test_output_dir is not None:
            # the predictions are saved as tiff files and then used for piexl-level evaluation in mvtec_ad_evaluation/evaluate_experiment.py
            # img_nm = os.path.split(path)[1].split('.')[0]
            # if not os.path.exists(os.path.join(test_output_dir, defect_class)):
            #     os.makedirs(os.path.join(test_output_dir, defect_class))
            # file = os.path.join(test_output_dir, defect_class, img_nm + '.tiff')
            # tifffile.imwrite(file, map_combined)

            # we save the originam image and anomaly maps for comparision.
            img_nm = os.path.split(path)[1].split('.')[0]
            if not os.path.exists(os.path.join(test_output_dir, defect_class)):
                os.makedirs(os.path.join(test_output_dir, defect_class))
            file = os.path.join(test_output_dir, defect_class, img_nm + '.png')

            fig, axs = plt.subplots(1, 2, figsize=(6, 6))
            axs[0].imshow(np.clip(image[0].cpu().numpy().transpose(1, 2, 0), 0, 1))
            axs[1].imshow(map_combined)
            axs[0].axis('off')
            axs[1].axis('off')

            plt.savefig(file, bbox_inches='tight', pad_inches=0)
            plt.close()

        y_true_image = 0 if defect_class == 'good' else 1
        y_score_image = np.max(map_combined)
        y_true.append(y_true_image)
        y_score.append(y_score_image)

    auc = roc_auc_score(y_true=y_true, y_score=y_score)
    # F1 score
    img_f1, img_threshold = calculate_f1_max(np.array(y_true), np.array(y_score))

    if config is not None: 
        # save image-wise info.
        for defect_type, image_id, y_true_image, y_score_image in zip(defect_types, image_ids, y_true, y_score):
            y_true_image_str = 'good' if y_true_image == 0 else 'anomalous'
            y_score_image_str = 'good' if y_score_image < img_threshold else 'anomalous'
            prediction_infos.append([defect_type, image_id, y_true_image_str, y_score_image_str, y_true_image, y_score_image])

        with open('predictions_' + config.subdataset + '.csv', 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerows(prediction_infos)

    return auc * 100, img_f1 * 100

@torch.no_grad()
def predict(image, teacher, student, autoencoder, teacher_mean, teacher_std,
            q_st_start=None, q_st_end=None, q_ae_start=None, q_ae_end=None):
    teacher_output = teacher(image)
    teacher_output = (teacher_output - teacher_mean) / teacher_std
    student_output = student(image)
    autoencoder_output = autoencoder(image)
    map_st = torch.mean((teacher_output - student_output[:, :out_channels])**2,
                        dim=1, keepdim=True)
    map_ae = torch.mean((autoencoder_output -
                         student_output[:, out_channels:])**2,
                        dim=1, keepdim=True)
    if q_st_start is not None:
        map_st = 0.1 * (map_st - q_st_start) / (q_st_end - q_st_start)
    if q_ae_start is not None:
        map_ae = 0.1 * (map_ae - q_ae_start) / (q_ae_end - q_ae_start)
    map_combined = 0.5 * map_st + 0.5 * map_ae
    return map_combined, map_st, map_ae

@torch.no_grad()
def map_normalization(validation_loader, teacher, student, autoencoder,
                      teacher_mean, teacher_std, desc='Map normalization'):
    maps_st = []
    maps_ae = []
    # ignore augmented ae image
    for image, _ in tqdm(validation_loader, desc=desc):
        if on_gpu:
            image = image.cuda()
        map_combined, map_st, map_ae = predict(
            image=image, teacher=teacher, student=student,
            autoencoder=autoencoder, teacher_mean=teacher_mean,
            teacher_std=teacher_std)
        maps_st.append(map_st)
        maps_ae.append(map_ae)
    maps_st = torch.cat(maps_st)
    maps_ae = torch.cat(maps_ae)
    q_st_start = torch.quantile(maps_st, q=0.9)
    q_st_end = torch.quantile(maps_st, q=0.995)
    q_ae_start = torch.quantile(maps_ae, q=0.9)
    q_ae_end = torch.quantile(maps_ae, q=0.995)
    return q_st_start, q_st_end, q_ae_start, q_ae_end

@torch.no_grad()
def teacher_normalization(teacher, train_loader):

    mean_outputs = []
    for train_image, _ in tqdm(train_loader, desc='Computing mean of features'):
        if on_gpu:
            train_image = train_image.cuda()
        teacher_output = teacher(train_image)
        mean_output = torch.mean(teacher_output, dim=[0, 2, 3])
        mean_outputs.append(mean_output)
    channel_mean = torch.mean(torch.stack(mean_outputs), dim=0)
    channel_mean = channel_mean[None, :, None, None]

    mean_distances = []
    for train_image, _ in tqdm(train_loader, desc='Computing std of features'):
        if on_gpu:
            train_image = train_image.cuda()
        teacher_output = teacher(train_image)
        distance = (teacher_output - channel_mean) ** 2
        mean_distance = torch.mean(distance, dim=[0, 2, 3])
        mean_distances.append(mean_distance)
    channel_var = torch.mean(torch.stack(mean_distances), dim=0)
    channel_var = channel_var[None, :, None, None]
    channel_std = torch.sqrt(channel_var)

    return channel_mean, channel_std

if __name__ == '__main__':
    main()
