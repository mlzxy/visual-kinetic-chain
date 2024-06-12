import cv2
import modules.data.kchain.base as base
import math
import common
from scipy.spatial.transform import Rotation
from PIL import Image
import numpy as np 
from copy import copy, deepcopy
from pathlib import Path
from common import load_pkl
import albumentations as A
import modules.data.augmentation as aug_module
import os


DEFAULT_ROBOT_P = {'panda': 1.0}

RLBENCH_BENCHMARK_TASKS = [
    "put_item_in_drawer",
    "reach_and_drag",
    "turn_tap",
    "slide_block_to_color_target",
    "open_drawer",
    "put_money_in_safe",
    "place_wine_at_rack_location",
    "sweep_to_dustpan_of_size",
    "meat_off_grill",
    "phone_on_base",
    "take_lid_off_saucepan",
    "close_microwave",
    "close_box",
    "basketball_in_hoop",
    "push_button", "pick_and_lift",
    "close_jar"
]



class RLBenchDataset(base.BaseVLDataset): # 20
    CONNECTIONS = {
        'panda':  [('Panda', 'Panda_link1_respondable'),
        ('Panda_link1_respondable', 'Panda_joint1'),
        ('Panda_joint1', 'Panda_link2_respondable'),
        ('Panda_link2_respondable', 'Panda_link3_respondable'),
        ('Panda_link3_respondable', 'Panda_joint4'),
        ('Panda_joint4', 'Panda_link4_respondable'),
        ('Panda_link4_respondable', 'Panda_link5_respondable'),
        ('Panda_link5_respondable', 'Panda_joint5'),
        ('Panda_joint5', 'Panda_link6_respondable'),
        ('Panda_link6_respondable', 'Panda_joint7'),
        ('Panda_joint7', 'Panda_link7_respondable'),
        ('Panda_link7_respondable', 'Panda_gripper'),
        ('Panda_gripper', 'Panda_rightfinger_respondable'),
        ('Panda_gripper', 'Panda_leftfinger_respondable')],
    
        'ur5': [('UR5', 'UR5_link2'),
        ('UR5_link2', 'UR5_joint2'),
        ('UR5_joint2', 'UR5_link3'),
        ('UR5_link3', 'UR5_joint3'),
        ('UR5_joint3', 'UR5_link4'),
        ('UR5_link4', 'UR5_link5'),
        ('UR5_link5', 'UR5_joint5'),
        ('UR5_joint5', 'UR5_link6'),
        ('UR5_link6', 'UR5_link7'),
        ('UR5_link7', 'ROBOTIQ_85'),
        ('ROBOTIQ_85', 'ROBOTIQ_85_RauxLink1Visible'),
        ('ROBOTIQ_85', 'ROBOTIQ_85_LauxLink1Visible'),
        ('ROBOTIQ_85_RauxLink1Visible', 'ROBOTIQ_85_RfingerTip'),
        ('ROBOTIQ_85_LauxLink1Visible', 'ROBOTIQ_85_LfingerTip')],

        'sawyer': [('Sawyer', 'Sawyer_link1'),
        ('Sawyer_link1', 'Sawyer_joint2'),
        ('Sawyer_joint2', 'Sawyer_link2'),
        ('Sawyer_link2', 'Sawyer_joint3'),
        ('Sawyer_joint3', 'Sawyer_link3'),
        ('Sawyer_link3', 'Sawyer_joint4'),
        ('Sawyer_joint4', 'Sawyer_link4'),
        ('Sawyer_link4', 'Sawyer_joint5'),
        ('Sawyer_joint5', 'Sawyer_link5'),
        ('Sawyer_link5', 'Sawyer_joint6'),
        ('Sawyer_joint6', 'Sawyer_link6'),
        ('Sawyer_link6', 'Sawyer_joint7'),
        ('Sawyer_joint7', 'Sawyer_link7'),
        ('Sawyer_link7', 'BaxterGripper'),
        ('BaxterGripper', 'BaxterGripper_rightPad_visible'),
        ('BaxterGripper', 'BaxterGripper_leftPad_visible')]
    }
    
    END_POINTS = {
        'panda': 'Panda_gripper',
        'ur5': 'ROBOTIQ_85',
        'sawyer': 'BaxterGripper'
    }

    SCENE_BOUNDS = [-0.5, -0.55, 0.75, 0.65, 0.5, 1.45]

    NUM_VIEWS=3
    
    @staticmethod
    def get_3d_2d_anchor_points(bounds, resolution=100):
        x_min, y_min, z_min, x_max, y_max, z_max = bounds
        array = np.array

        R, T, intrinsics, image_sizes = [{'c0': array([[ 0.99580104,  0.07454367,  0.0531364 ],
                [-0.00807397, -0.50666924,  0.86210292],
                [ 0.09118688, -0.85891201, -0.50393987]]),
        'c1': array([[ 0.8928785 , -0.39769636, -0.21120031],
                [-0.0667766 , -0.58077933,  0.81131737],
                [-0.44531884, -0.7103046 , -0.54512217]]),
        'c3': array([[-0.77700172, -0.62510661,  0.07421515],
                [-0.35085313,  0.52793248,  0.7734266 ],
                [-0.52265495,  0.57491603, -0.62952461]]),
        'front': array([[ 8.72275344e-07, -9.99999881e-01, -6.68623797e-07],
                [-6.42787582e-01, -8.32382456e-07,  7.66044281e-01],
                [-7.66044281e-01, -1.57439685e-07, -6.42787463e-01]])},
        {'c0': array([-0.17695284, -0.89997483,  2.01099815]),
        'c1': array([-0.01699888, -0.79623299,  1.89577638]),
        'c3': array([ 0.06581596, -0.65879423,  1.93515907]),
        'front': array([ 8.35648835e-08, -5.34556148e-01,  2.21084547e+00])},
        {'c0': array([[-351.6771208,    0.       ,  128.       ],
                [   0.       , -351.6771208,  128.       ],
                [   0.       ,    0.       ,    1.       ]]),
        'c1': array([[-351.6771208,    0.       ,  128.       ],
                [   0.       , -351.6771208,  128.       ],
                [   0.       ,    0.       ,    1.       ]]),
        'c3': array([[-351.6771208,    0.       ,  128.       ],
                [   0.       , -351.6771208,  128.       ],
                [   0.       ,    0.       ,    1.       ]]),
        'front': array([[-351.6771208,    0.       ,  128.       ],
                [   0.       , -351.6771208,  128.       ],
                [   0.       ,    0.       ,    1.       ]])},
        {'c0': [256, 256], 'c1': [256, 256], 'c3': [256, 256], 'front': [256, 256]}]

            
        x = torch.linspace(x_min, x_max, resolution)
        y = torch.linspace(y_min, y_max, resolution)
        z = torch.linspace(z_min, z_max, resolution)
        anchor_points3d = torch.cartesian_prod(x, y, z).numpy()

        points2d_from_3d = {
            cam: cv2.projectPoints(anchor_points3d,  
                    R[cam],  T[cam], intrinsics[cam], None)[0].reshape(-1, 2)
            for cam in ['c0', 'c1', 'c3', 'front']
        }

        return anchor_points3d, points2d_from_3d



    def __init__(self, path, robot_p=None, transform=None, limits=1000, views=None,
                camera=None, key_frame=False, point2d_as_action=False, point3d_as_action=False,
                horizon=30, min_points_per_edge=20, ordered=False, **kwargs):
        super().__init__(transform, limits, horizon, min_points_per_edge, ordered, **kwargs)
        self.robot_p = copy(DEFAULT_ROBOT_P)
        self.robot_p.update(robot_p or {})
        self.path = Path(path)
        self.views = views
        self.RLBENCH_BENCHMARK_TASKS = RLBENCH_BENCHMARK_TASKS
        self.all_tasks = {
            'panda': self.RLBENCH_BENCHMARK_TASKS,
        }
        self.camera = camera
        self.key_frame = key_frame
        self.point2d_as_action = point2d_as_action
        self.point3d_as_action = point3d_as_action
        if self.point3d_as_action: 
            self.point2d_as_action = False
            self.anchor_3d_points, _ = self.get_3d_2d_anchor_points(self.SCENE_BOUNDS)

        self.num_episodes = {}
        for robot in self.all_tasks:
            if self.robot_p[robot] == 0: continue
            self.num_episodes[robot] = {}
            for t in (self.path / robot).iterdir():
                if t.is_dir() and t.name in self.RLBENCH_BENCHMARK_TASKS:
                    L = len(list((t / 'episodes').iterdir()))
                    if L > 0:
                        self.num_episodes[robot][t.name] = L
                
        self.camera_parameters = None

    def get(self, index, state=None) -> base.VLSample:
        rgb_others = []
        task, robot, episode, frame_ids, cam = index
        fid = int(frame_ids[0])
        episode_path = self.path / robot / task / 'episodes' / f'episode{episode}'
        if state is None: state = load_pkl(episode_path / 'low_dim_obs.pkl')
        desc = load_pkl(episode_path / 'variation_descriptions.pkl')
        if self.debug: print(index, desc)
        desc_ind = 0 
        desc = desc[desc_ind]
        desc_emb = self.load_lang_emb(episode_path, desc_ind) 

        joints, gripper, xyz, rpy, skel_pts, end_points, action = [], [], [], [], [], [], []
        rgb = np.array(Image.open(episode_path / f'{cam}_rgb' / f'{int(frame_ids[0])}.jpg'))
        P = 512

        if self.include_other_frames == 'last':
            rgb_others.append(np.array(Image.open(episode_path / f'{cam}_rgb' / f'{int(frame_ids[-1])}.jpg')))
        elif self.include_other_frames == 'all':
            for _fid in frame_ids[1:]:
                rgb_others.append(np.array(Image.open(episode_path / f'{cam}_rgb' / f'{int(_fid)}.jpg')))

        if self.key_frame:
            frame_ids = [int(frame_ids[-1]), ]
        
        sparse_skeletons = []
        
        for frame_id in frame_ids:
            l_frame_id, r_frame_id = int(math.floor(frame_id)), int(math.ceil(frame_id))
            if l_frame_id == r_frame_id:
                l_w, r_w = 1, 0
            else:
                l_w, r_w = 1 - (frame_id - l_frame_id), 1 - (r_frame_id - frame_id)

            l_step = deepcopy(state['steps'][l_frame_id])
            l_skeleton_data = deepcopy(l_step['skeleton_data'])
            r_step = deepcopy(state['steps'][r_frame_id])
            r_skeleton_data = deepcopy(r_step['skeleton_data'])

            l_joint_pts2d = l_skeleton_data['points_2d'][f'{cam}_camera']['joints']
            l_link_pts2d = l_skeleton_data['points_2d'][f'{cam}_camera']['links']
            r_joint_pts2d = r_skeleton_data['points_2d'][f'{cam}_camera']['joints']
            r_link_pts2d = r_skeleton_data['points_2d'][f'{cam}_camera']['links']
            
            l_joint_names = l_skeleton_data['joints']
            l_link_names = l_skeleton_data['links']
            r_joint_names = r_skeleton_data['joints']
            r_link_names = r_skeleton_data['links']
            if l_joint_names == r_joint_names and l_link_names == r_link_names:
                # need: `step`, `joint_pts2d`, `link_pts2d`
                joint_names = l_joint_names
                link_names = l_link_names

                joint_pts2d = l_w * l_joint_pts2d + r_w * r_joint_pts2d
                link_pts2d = l_w * l_link_pts2d + r_w * r_link_pts2d

                step = {
                    'joint_positions': l_w * l_step['joint_positions'] + r_w * r_step['joint_positions'],

                    'gripper_pose': common.cat([l_w * l_step['gripper_pose'][:3] + r_w * r_step['gripper_pose'][:3], 
                                            common.interpolate_quat(l_w, l_step['gripper_pose'][3:], r_w, r_step['gripper_pose'][3:])]),

                    'gripper_open':  r_step['gripper_open'] if l_w < r_w else l_step['gripper_open'],
                    'ignore_collisions':  r_step['ignore_collisions'] if l_w < r_w else l_step['ignore_collisions']
                }
            else:
                joint_pts2d = l_joint_pts2d if l_w > r_w else r_joint_pts2d
                link_pts2d = l_link_pts2d if l_w > r_w else r_link_pts2d
                step = l_step if l_w > r_w else r_step
                joint_names = l_joint_names if l_w > r_w else r_joint_names
                link_names = l_link_names if l_w > r_w else r_link_names            

            end_points.append(link_pts2d[link_names.index(self.END_POINTS[robot])].flatten() * 224 / len(rgb))

            points, edges = base.obtain_skeletons(np.concatenate([link_pts2d, joint_pts2d]), link_names + joint_names, self.CONNECTIONS[robot])
            sparse_skeletons.append([np.copy(points), edges])
            points += (P // 2)
            skel_pts.append([points, edges])

            joints.append(step['joint_positions'])
            gripper.append(np.array([step['gripper_open'], step['ignore_collisions']]))
            xyz.append(step['gripper_pose'][:3])
            quat_xyzw = step['gripper_pose'][3:]
            r = Rotation.from_quat(quat_xyzw)
            euler = base.sensitive_gimble_fix(r.as_euler("xyz", degrees=True))
            
            if self.point2d_as_action:
                assert self.key_frame
                rpy.append(common.discretize_euler(euler))
                if self.camera_parameters is None:
                    self.camera_parameters = {}
                    for _cam in ['c0', 'c1', 'c3', 'front']:
                        E = np.linalg.inv(l_step[_cam + '_camera_extrinsics'])
                        self.camera_parameters[_cam] = {
                            'R': E[:3, :3], 'T': E[:3, 3], 'Intrinsic': l_step[_cam + '_camera_intrinsics'],
                        }
                screen_pts, _ = cv2.projectPoints(xyz[-1].reshape(-1, 3),  
                                            self.camera_parameters[cam]['R'],  self.camera_parameters[cam]['T'], 
                                            self.camera_parameters[cam]['Intrinsic'], None)
                action.append(screen_pts.flatten() * 224 / 256) # assuming the recorded data is 256x256, and we want 224x224
            elif self.point3d_as_action:
                dist = np.linalg.norm(self.anchor_3d_points - xyz[-1].reshape(-1, 3), axis=1)
                action.append(dist.argmin())
            else:
                euler = np.deg2rad(euler)
                rpy.append(euler)
                rpy_normed = base.normalize_rad(rpy[-1])
                action.append(common.cat([xyz[-1], rpy_normed, gripper[-1]], 0))

            
        skel_pts = self.combine_skeletons_in_horizon(skel_pts, rgb.shape[:2], P)
        rgb, skel_pts = self.transform(rgb, skel_pts)

        self.debug_visualize(rgb, skel_pts)

        return base.VLSample(
            env="rlbench",
            task=task,
            episode=episode,
            episode_path=episode_path,
            frame_id=fid, lang=desc + f' [{cam}]', lang_emb=desc_emb, 
            rgb=rgb, skel_pts=skel_pts, 
            joints=np.stack(joints), gripper=np.stack(gripper),
            xyz=np.stack(xyz), rpy=np.stack(rpy), end_points=np.stack(end_points),
            action=np.stack(action),
            skel_hms=self.render_skel_hms(rgb, skel_pts),
            sample_id=self.get_sample_id('rlbench', 100 * self.RLBENCH_BENCHMARK_TASKS.index(task) + episode, fid),
            skeleton=sparse_skeletons,
            rgb_others=rgb_others
        )
    
    
    def __getitem__(self, ind) -> base.VLSample:
        robots = [k for k in self.robot_p]
        robot_probs = [self.robot_p[k] for k in robots]
        robot = np.random.choice(robots, p=robot_probs)
        task = np.random.choice(list(self.num_episodes[robot].keys()))
        episode = np.random.randint(0, self.num_episodes[robot][task]) 
        episode_path = self.path / robot / task / 'episodes' / f'episode{episode}'
        state = load_pkl(episode_path / 'low_dim_obs.pkl')
        kfs = common.load_json(episode_path / 'kf.json')
        if 1 in kfs: kfs.remove(1)
        if robot == 'ur5':
            kfs = [k for k in kfs if k >= 30]
        kfs = [0] + kfs
        kf = np.random.choice(kfs[:-1])
        next_kf = base.query_next_kf(kf, kfs)
        if self.camera is None:
            cam = np.random.choice(['c0', 'c1', 'front', 'c3'])
        else:
            cam = self.camera
        if self.key_frame:
            timesteps = [kf, next_kf]
        else:
            timesteps = np.linspace(kf, next_kf, self.horizon)
        
        if self.multi_view:
            if self.views:
                cams = self.views
            else:
                cams = np.random.choice(['c0', 'c1', 'front', 'c3'], size=self.NUM_VIEWS, replace=False)
            return [self.get([task, robot, episode, timesteps, cam], state=state) for cam in cams]
        else:
            return self.get([task, robot, episode, timesteps, cam], state=state)
        

    

class CalvinDataset(base.BaseVLDataset): # 30
    CONNECTIONS = [('panda_link0', 'panda_link1'),
                ('panda_link1', 'panda_link2'),
                ('panda_link2', 'panda_link3'),
                ('panda_link3', 'panda_link4'),
                ('panda_link4', 'panda_link5'),
                ('panda_link5', 'panda_link7'),
                ('panda_link7', 'panda_link8'),
                ('panda_link8', 'panda_rightfinger'),
                ('panda_link8', 'panda_leftfinger'),
                ('panda_rightfinger', 'finger_right_tip'),
                ('panda_leftfinger', 'finger_left_tip') ]
    
    END_POINTS = 'panda_link8'
    NUM_VIEWS = 1
    
    def __init__(self, path, transform=None, limits=1000, 
                 horizon=30, min_points_per_edge=20, ordered=False, random=False,
                 state_as_action=True, **kwargs):
        super().__init__(transform, limits, horizon, min_points_per_edge, ordered=False, **kwargs)
        self.path = Path(path)
        self.num_episodes = len([a for a in self.path.iterdir() if a.is_dir()])
        self.random = random
    
        
    def __getitem__(self, ind) -> base.VLSample:
        episode_id = np.random.randint(0, self.num_episodes)
        state = load_pkl(self.path / f'episode_{str(episode_id).zfill(7)}' / 'low_dim.pkl')
        num_frames = len(state['states'])
        if self.random:
            start_fid = np.random.randint(0, max(1, num_frames - self.horizon))
            timesteps = common.subsample_or_pad(list(range(start_fid, num_frames)), self.horizon)
            return self.get([episode_id, timesteps], state=state)
        else:
            timesteps = common.subsample_or_pad(list(range(num_frames)), self.horizon)
            return self.get([episode_id, timesteps], state=state)
    
    def get(self, ind, state=None) -> base.VLSample:
        episode_id, frame_ids = ind
        rgb_others = []
        fid = int(frame_ids[0])
        episode_path = self.path / f'episode_{str(episode_id).zfill(7)}'
        num_frames = len(state['states'])
        if state is None: state = load_pkl(episode_path / 'low_dim.pkl')
        task = state['task']
        desc = state['language']
        if self.debug: print(episode_id, fid, desc)
        desc_emb = self.load_lang_emb(episode_path) 
        
        rgb = np.array(Image.open(episode_path / f'rgb' / f'{str(fid).zfill(7)}.jpg'))
        rgb, (top_pad, left_pad) = base.center_pad_image(rgb, return_pad=True)
        P = 512

        if self.include_other_frames == 'last':
            _rgb = np.array(Image.open(episode_path / f'rgb' / f'{str(frame_ids[-1]).zfill(7)}.jpg'))
            _rgb, _ = base.center_pad_image(_rgb, return_pad=True)
            rgb_others.append(_rgb)
        elif self.include_other_frames == 'all':
            for _fid in frame_ids[1:]:
                _rgb = np.array(Image.open(episode_path / f'rgb' / f'{str(_fid).zfill(7)}.jpg'))
                _rgb, _ = base.center_pad_image(_rgb, return_pad=True)
                rgb_others.append(_rgb)


        joints, gripper, xyz, rpy, skel_pts, end_points, action = [], [], [], [], [], [], []
        
        link_names = state['meta_data']['link_names']
        sparse_skeletons = []
        for frame_id in frame_ids:
            robot_obs = deepcopy(state['states'][frame_id])
            skeletons = deepcopy(state['skeletons'][frame_id])
            
            joints.append(robot_obs[[7, 8, 9, 10, 11, 12, 13]])
            gripper.append(np.array([robot_obs[6] / 2]))
            xyz.append(robot_obs[:3])
            rpy.append(robot_obs[3:6])

            skeletons[:, 0] += left_pad
            skeletons[:, 1] += top_pad

            end_points.append(skeletons[link_names.index(self.END_POINTS)] * 224 / len(rgb))

            points, edges = base.obtain_skeletons(skeletons, link_names, self.CONNECTIONS)
            sparse_skeletons.append([np.copy(points) * 256 / len(rgb), edges])
            points += (P // 2)
            skel_pts.append([points, edges])

            prev_frame_id = int(round(max(frame_id - 1, 0)))
            act_vec = state['actions'][prev_frame_id]['abs']               

            act = np.zeros([7], dtype=np.float32)
            act[:6] = robot_obs[:6]
            act[3:6] = base.normalize_rad(act[3:6])
            act[-1] = (act_vec[-1] + 1) / 2

            action.append(act)

        skel_pts = self.combine_skeletons_in_horizon(skel_pts, rgb.shape[:2], P)
        rgb, skel_pts = self.transform(rgb, skel_pts)

        self.debug_visualize(rgb, skel_pts)

        return base.VLSample(
            env="calvin",
            task=task,
            episode=episode_id,
            episode_path=episode_path,
            frame_id=fid, lang=desc, lang_emb=desc_emb, 
            rgb=rgb, skel_pts=skel_pts, 
            joints=np.stack(joints), gripper=np.stack(gripper),
            xyz=np.stack(xyz), rpy=np.stack(rpy), end_points=np.stack(end_points),
            action=np.stack(action),
            skel_hms=self.render_skel_hms(rgb, skel_pts),
            sample_id=self.get_sample_id('calvin', episode_id, fid),
            skeleton=sparse_skeletons,
            rgb_others=rgb_others
        )
    
    


if __name__ == "__main__":
    pass