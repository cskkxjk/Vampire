import mmcv
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from tqdm import tqdm
import json

def generate_info(nusc, scenes, max_cam_sweeps=6, max_lidar_sweeps=10, occ_anno=None):
    infos = list()
    for cur_scene in tqdm(nusc.scene):
        if cur_scene['name'] not in scenes:
            continue
        first_sample_token = cur_scene['first_sample_token']
        cur_sample = nusc.get('sample', first_sample_token)
        while True:
            info = dict()
            sweep_cam_info = dict()
            cam_datas = list()
            lidar_datas = list()
            info['sample_token'] = cur_sample['token']
            info['timestamp'] = cur_sample['timestamp']
            info['scene_token'] = cur_sample['scene_token']
            cam_names = [
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'
            ]
            lidar_names = ['LIDAR_TOP']
            cam_infos = dict()
            lidar_infos = dict()
            occ_infos = dict()
            for cam_name in cam_names:
                cam_data = nusc.get('sample_data',
                                    cur_sample['data'][cam_name])
                cam_datas.append(cam_data)
                sweep_cam_info = dict()
                sweep_cam_info['sample_token'] = cam_data['sample_token']
                sweep_cam_info['ego_pose'] = nusc.get(
                    'ego_pose', cam_data['ego_pose_token'])
                sweep_cam_info['timestamp'] = cam_data['timestamp']
                sweep_cam_info['is_key_frame'] = cam_data['is_key_frame']
                sweep_cam_info['height'] = cam_data['height']
                sweep_cam_info['width'] = cam_data['width']
                sweep_cam_info['filename'] = cam_data['filename']
                sweep_cam_info['calibrated_sensor'] = nusc.get(
                    'calibrated_sensor', cam_data['calibrated_sensor_token'])
                cam_infos[cam_name] = sweep_cam_info
            for lidar_name in lidar_names:
                lidar_data = nusc.get('sample_data',
                                      cur_sample['data'][lidar_name])
                lidar_datas.append(lidar_data)
                sweep_lidar_info = dict()
                sweep_lidar_info['sample_token'] = lidar_data['sample_token']
                # lidar token to save results for submition
                sweep_lidar_info['lidar_token'] = lidar_data['token']
                sweep_lidar_info['ego_pose'] = nusc.get(
                    'ego_pose', lidar_data['ego_pose_token'])
                sweep_lidar_info['timestamp'] = lidar_data['timestamp']
                sweep_lidar_info['filename'] = lidar_data['filename']
                sweep_lidar_info['calibrated_sensor'] = nusc.get(
                    'calibrated_sensor', lidar_data['calibrated_sensor_token'])
                # add lidarseg labels
                try:
                    sweep_lidar_info['lidarseg_labels_filename'] = nusc.get(
                        'lidarseg', lidar_data['token'])['filename']
                except KeyError:
                    sweep_lidar_info['lidarseg_labels_filename'] = None
                lidar_infos[lidar_name] = sweep_lidar_info
            lidar_sweeps = [dict() for _ in range(max_lidar_sweeps)]
            cam_sweeps = [dict() for _ in range(max_cam_sweeps)]
            info['cam_infos'] = cam_infos
            info['lidar_infos'] = lidar_infos
            if occ_anno is not None:
                occ_infos['occ_gt_path'] = occ_anno['scene_infos'][cur_scene['name']][cur_sample['token']]['gt_path']
            info['occ_infos'] = occ_infos
            # for i in range(max_cam_sweeps):
            #     cam_sweeps.append(dict())
            for k, cam_data in enumerate(cam_datas):
                sweep_cam_data = cam_data
                for j in range(max_cam_sweeps):
                    if sweep_cam_data['prev'] == '':
                        break
                    else:
                        sweep_cam_data = nusc.get('sample_data',
                                                  sweep_cam_data['prev'])
                        sweep_cam_info = dict()
                        sweep_cam_info['sample_token'] = sweep_cam_data[
                            'sample_token']
                        if sweep_cam_info['sample_token'] != cam_data[
                                'sample_token']:
                            break
                        sweep_cam_info['ego_pose'] = nusc.get(
                            'ego_pose', cam_data['ego_pose_token'])
                        sweep_cam_info['timestamp'] = sweep_cam_data[
                            'timestamp']
                        sweep_cam_info['is_key_frame'] = sweep_cam_data[
                            'is_key_frame']
                        sweep_cam_info['height'] = sweep_cam_data['height']
                        sweep_cam_info['width'] = sweep_cam_data['width']
                        sweep_cam_info['filename'] = sweep_cam_data['filename']
                        sweep_cam_info['calibrated_sensor'] = nusc.get(
                            'calibrated_sensor',
                            cam_data['calibrated_sensor_token'])
                        cam_sweeps[j][cam_names[k]] = sweep_cam_info

            for k, lidar_data in enumerate(lidar_datas):
                sweep_lidar_data = lidar_data
                for j in range(max_lidar_sweeps):
                    if sweep_lidar_data['prev'] == '':
                        break
                    else:
                        sweep_lidar_data = nusc.get('sample_data',
                                                    sweep_lidar_data['prev'])
                        sweep_lidar_info = dict()
                        sweep_lidar_info['sample_token'] = sweep_lidar_data[
                            'sample_token']
                        if sweep_lidar_info['sample_token'] != lidar_data[
                                'sample_token']:
                            break
                        sweep_lidar_info['ego_pose'] = nusc.get(
                            'ego_pose', sweep_lidar_data['ego_pose_token'])
                        sweep_lidar_info['timestamp'] = sweep_lidar_data[
                            'timestamp']
                        sweep_lidar_info['is_key_frame'] = sweep_lidar_data[
                            'is_key_frame']
                        sweep_lidar_info['filename'] = sweep_lidar_data[
                            'filename']
                        sweep_lidar_info['calibrated_sensor'] = nusc.get(
                            'calibrated_sensor',
                            lidar_data['calibrated_sensor_token'])
                        try:
                            sweep_lidar_info['lidarseg_labels_filename'] = nusc.get(
                                'lidarseg', sweep_lidar_data['token'])['filename']
                        except KeyError:
                            sweep_lidar_info['lidarseg_labels_filename'] = None
                        lidar_sweeps[j][lidar_names[k]] = sweep_lidar_info
            # Remove empty sweeps.
            for i, sweep in enumerate(cam_sweeps):
                if len(sweep.keys()) == 0:
                    cam_sweeps = cam_sweeps[:i]
                    break
            for i, sweep in enumerate(lidar_sweeps):
                if len(sweep.keys()) == 0:
                    lidar_sweeps = lidar_sweeps[:i]
                    break
            info['cam_sweeps'] = cam_sweeps
            info['lidar_sweeps'] = lidar_sweeps
            ann_infos = list()
            if 'anns' in cur_sample:
                for ann in cur_sample['anns']:
                    ann_info = nusc.get('sample_annotation', ann)
                    velocity = nusc.box_velocity(ann_info['token'])
                    if np.any(np.isnan(velocity)):
                        velocity = np.zeros(3)
                    ann_info['velocity'] = velocity
                    ann_infos.append(ann_info)
                info['ann_infos'] = ann_infos
            infos.append(info)
            if cur_sample['next'] == '':
                break
            else:
                cur_sample = nusc.get('sample', cur_sample['next'])
    return infos


def main():
    trainval_nusc = NuScenes(version='v1.0-trainval',
                             dataroot='./data/nuScenes/',
                             verbose=True)
    train_scenes = splits.train
    val_scenes = splits.val
    # # vis_scenes = ['scene-0916']
    # scene_0012_0018 = ['scene-0012', 'scene-0013', 'scene-0014', 'scene-0015', 'scene-0016', 'scene-0017', 'scene-0018']
    # scene_0904_0917 = ['scene-0904', 'scene-0905', 'scene-0906', 'scene-0907', 'scene-0908', 'scene-0909', 'scene-0910',
    #                    'scene-0911', 'scene-0912', 'scene-0913', 'scene-0914', 'scene-0915', 'scene-0916', 'scene-0917']
    trainval_scenes = train_scenes + val_scenes
    occ_anno_path = './data/nuScenes/annotations.json'
    with open(occ_anno_path, 'r') as f:
        occ_anno = json.load(f)
    # # vis_infos = generate_info(trainval_nusc, vis_scenes, occ_anno=occ_anno)
    # scene_0012_0018_infos = generate_info(trainval_nusc, scene_0012_0018, occ_anno=occ_anno)
    # scene_0904_0917_infos = generate_info(trainval_nusc, scene_0904_0917, occ_anno=occ_anno)
    # mmcv.dump(scene_0012_0018_infos, './data/nuScenes/nuscenes_occ_infos_scene_0012_0018.pkl')
    # mmcv.dump(scene_0904_0917_infos, './data/nuScenes/nuscenes_occ_infos_scene_0904_0917.pkl')
    # # mmcv.dump(vis_infos, './data/nuScenes/nuscenes_occ_infos_vis.pkl')
    train_infos = generate_info(trainval_nusc, train_scenes, occ_anno=occ_anno)
    val_infos = generate_info(trainval_nusc, val_scenes, occ_anno=occ_anno)
    trainval_infos = generate_info(trainval_nusc, trainval_scenes, occ_anno=occ_anno)
    mmcv.dump(train_infos, './data/nuScenes/nuscenes_occ_infos_train.pkl')
    mmcv.dump(val_infos, './data/nuScenes/nuscenes_occ_infos_val.pkl')
    mmcv.dump(trainval_infos, './data/nuScenes/nuscenes_occ_infos_trainval.pkl')
    # test_nusc = NuScenes(version='v1.0-test',
    #                      dataroot='./data/nuScenes/',
    #                      verbose=True)
    # test_scenes = splits.test
    # test_infos = generate_info(test_nusc, test_scenes)
    # mmcv.dump(test_infos, './data/nuScenes/nuscenes_infos_test.pkl')


if __name__ == '__main__':
    main()
