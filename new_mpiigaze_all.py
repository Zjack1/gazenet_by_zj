#!/usr/bin/env python

import argparse
import pathlib

import cv2
import h5py
import numpy as np
import pandas as pd
import scipy.io
import tqdm
from math import cos, sin
label_outpath = "../MPIIGaze/new_data/right_img_all_label.txt"
outfile = open(label_outpath, 'w')

def draw_eye_line(img, yaw, pitch, size=60):
    tdx = 30
    tdy = 18

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (-sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (0, 255, 0), 2)

    return img


def convert_pose(vector: np.ndarray) -> np.ndarray:
    rot = cv2.Rodrigues(np.array(vector).astype(np.float32))[0]
    vec = rot[:, 2]
    pitch = np.arcsin(vec[1])
    yaw = np.arctan2(vec[0], vec[2])
    return np.array([pitch, yaw]).astype(np.float32)


def convert_gaze(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    pitch = np.arcsin(-y)
    yaw = np.arctan2(-x, -z)
    return np.array([pitch, yaw]).astype(np.float32)


def get_eval_info(person_id: str, eval_dir: pathlib.Path) -> pd.DataFrame:
    eval_path = eval_dir / f'{person_id}.txt'
    df = pd.read_csv(eval_path,
                     delimiter=' ',
                     header=None,
                     names=['path', 'side'])
    df['day'] = df.path.apply(lambda path: path.split('/')[0])
    df['filename'] = df.path.apply(lambda path: path.split('/')[1])
    df = df.drop(['path'], axis=1)
    return df


def save_one_person(person_id: str, data_dir: pathlib.Path, eval_dir: pathlib.Path):

    left_images = dict()
    left_poses = dict()
    left_gazes = dict()
    right_images = dict()
    right_poses = dict()
    right_gazes = dict()
    filenames = dict()
    person_dir = data_dir / person_id
    for path in sorted(person_dir.glob('*')):  # 返回一个某一种文件夹下面的某一类型文件路径列表
        mat_data = scipy.io.loadmat(path.as_posix(),
                                    struct_as_record=False,
                                    squeeze_me=True)
        data = mat_data['data']

        day = path.stem
        left_images[day] = data.left.image
        left_poses[day] = data.left.pose
        left_gazes[day] = data.left.gaze
        # a = data.left.image[1]
        # b = data.left.gaze[1]
        # cv2.imwrite("./333.jpg", a)


        right_images[day] = data.right.image
        # a = data.right.image[1]
        # cv2.imwrite("./2.jpg",a)
        right_poses[day] = data.right.pose
        right_gazes[day] = data.right.gaze

        filenames[day] = mat_data['filenames']

        if not isinstance(filenames[day], np.ndarray):
            left_images[day] = np.array([left_images[day]])
            left_poses[day] = np.array([left_poses[day]])
            left_gazes[day] = np.array([left_gazes[day]])
            right_images[day] = np.array([right_images[day]])
            right_poses[day] = np.array([right_poses[day]])
            right_gazes[day] = np.array([right_gazes[day]])
            filenames[day] = np.array([filenames[day]])

    df = get_eval_info(person_id, eval_dir)
    paths = []
    images = []
    poses = []
    gazes = []
    for _, row in df.iterrows():
        day = row.day
        index = np.where(filenames[day] == row.filename)[0][0]
        if row.side == 'left':  # 非镜像，图像的左边就是真实的右边
            image = left_images[day][index]
            pose_angle_rad = convert_pose(left_poses[day][index])
            gaze_angle_rad = convert_gaze(left_gazes[day][index])
            # 命名是按照镜像来的
            img_path = "../MPIIGaze/new_data/right_img_all/" + str(person_id) + "_" + str(day) + "_" + str(index) + "_right.jpg"
            save_img_path = "MPIIGaze/new_data/right_img_all/" + str(person_id) + "_" + str(day) + "_" + str(index) + "_right.jpg"
            cv2.imwrite(img_path, image)

            gaze_angle_rad_save = " ".join(gaze_angle_rad.astype("str"))
            pose_angle_rad_save = " ".join(pose_angle_rad.astype("str"))
            left_gazes_vec_save = " ".join(left_gazes[day][index].astype("str"))
            left_poses_vec_save = " ".join(left_poses[day][index].astype("str"))
            save_str = save_img_path + " " + gaze_angle_rad_save + " " + pose_angle_rad_save + " " + left_gazes_vec_save + " " + left_poses_vec_save
            outfile.write(save_str + "\n")
        else:
            image = right_images[day][index][:, ::-1]
            pose_angle_rad = convert_pose(right_poses[day][index]) * np.array([1, -1])
            gaze_angle_rad = convert_gaze(right_gazes[day][index]) * np.array([1, -1])

            # 命名是按照镜像来的
            img_path = "../MPIIGaze/new_data/right_img_all/" + str(person_id) + "_" + str(day) + "_" + str(
                index) + "_right_mirror.jpg"
            save_img_path = "MPIIGaze/new_data/right_img_all/" + str(person_id) + "_" + str(day) + "_" + str(
                index) + "_right_mirror.jpg"
            cv2.imwrite(img_path, image)

            gaze_angle_rad_save = " ".join(gaze_angle_rad.astype("str"))
            pose_angle_rad_save = " ".join(pose_angle_rad.astype("str"))
            left_gazes_vec_save = " ".join(right_poses[day][index].astype("str"))
            left_poses_vec_save = " ".join(right_poses[day][index].astype("str"))
            save_str = save_img_path + " " + gaze_angle_rad_save + " " + pose_angle_rad_save + " " + left_gazes_vec_save + " " + left_poses_vec_save
            outfile.write(save_str + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="../MPIIGaze")
    args = parser.parse_args()

    dataset_dir = pathlib.Path(args.dataset)
    for person_id in tqdm.tqdm(range(15)):
        person_id = f'p{person_id:02}'
        data_dir = dataset_dir / 'Data' / 'Normalized'
        eval_dir = dataset_dir / 'Evaluation Subset' / 'sample list for eye image all'
        save_one_person(person_id, data_dir, eval_dir)


if __name__ == '__main__':
    main()
