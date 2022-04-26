import sys,os,imageio

from opt import config_parser
from data import dataset_dict
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# models
from models import *
from renderer import *
from data.ray_utils import get_rays
from scipy.spatial.transform import Rotation as R

from tqdm import tqdm


from skimage.metrics import structural_similarity

# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer, loggers


from data.ray_utils import ray_marcher

def decode_batch(batch):
    rays = batch['rays']  # (B, 8)
    rgbs = batch['rgbs']  # (B, 3)
    return rays, rgbs

def unpreprocess(data, shape=(1,1,3,1,1)):
    # to unnormalize image for visualization
    # data N V C H W
    device = data.device
    mean = torch.tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]).view(*shape).to(device)
    std = torch.tensor([1 / 0.229, 1 / 0.224, 1 / 0.225]).view(*shape).to(device)

    return (data - mean) / std


def normalize(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.eye(4)
    m[:3] = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3,:3].T, (pts-c2w[:3,3])[...,np.newaxis])[...,0]
    return tt

def poses_avg(poses):

    center = poses[:, :3, 3].mean(0)
    print('center in poses_avg', center)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = viewmatrix(vec2, up, center)
    
    return c2w

def render_path_imgs(c2ws_all, focal):
    T = c2ws_all[...,3]

    return render_poses

def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, N_rots=2, N=120):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    
    for theta in np.linspace(0., 2. * np.pi * N_rots, N+1)[:-1]:
    # for theta in np.linspace(0., 5, N+1)[:-1]:
        # spiral
        # c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads)

        # 关于使用平均姿态的相关问题
        # 从训练集推算出来的平均姿态方向基本平行于z轴，因为训练集中大多数图片是正面的，
        # 但是存在一个问题，将z和平均姿态相乘后得到的方向也基本上和z平行，所以无论怎么调整看起来都是平行的，
        # 别用平均姿态看其他位置的照片，直接用世界坐标系即可！！！！
        # 但是需要用别的姿态大致估计一下位置参数
        c = np.array([(np.cos(theta)*theta)/10, (-np.sin(theta)*theta)/10, -0.1]) 

        # 这个是因为作者在读取并规范化相机姿态的时候作了poses*blender2opencv，转换了坐标系，
        # 我用的数据无需转换，但是这里加个负号就解决了，目前不影响什么，记住就行
        z = -(normalize(c - np.array([0,0,-focal])))
        print("c", c)
        print("z", z)
        render_poses.append(viewmatrix(z, up, c))
    return render_poses

def get_spiral(c2ws_all, near_far, rads_scale=0.5, N_views=120):

    # center pose
    c2w = poses_avg(c2ws_all)
    print('poses_avg', c2w)
    
    # Get average pose
    up = normalize(c2ws_all[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset
    close_depth, inf_depth = near_far
    print('near and far bounds', close_depth, inf_depth)
    dt = .75
    mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
    focal = mean_dz
    print(focal)

    # Get radii for spiral path
    shrink_factor = .8
    zdelta = close_depth * .2
    tt = c2ws_all[:,:3,3] - c2w[:3,3][None]
    rads = np.percentile(np.abs(tt), 70, 0)*rads_scale
    print("rads",rads)
    render_poses = render_path_spiral(c2w, up, rads, focal, zdelta, zrate=.5, N=N_views)
    return np.stack(render_poses)

def position2angle(position, N_views=16, N_rots = 2):
    ''' nx3 '''
    position = normalize(position)
    theta = np.arccos(position[:,2])/np.pi*180
    phi = np.arctan2(position[:,1],position[:,0])/np.pi*180
    return [theta,phi]

def pose_spherical_nerf(euler, radius=0.01):
    c2ws_render = np.eye(4)
    c2ws_render[:3,:3] =  R.from_euler('xyz', euler, degrees=True).as_matrix()
    # 保留旋转矩阵的最后一列再乘个系数就能当作位置？
    c2ws_render[:3,3]  = c2ws_render[:3,:3] @ np.array([0.0,0.0,-radius])
    return c2ws_render

def create_spheric_poses(radius, n_poses=120):
    """
    Create circular poses around z axis.
    Inputs:
        radius: the (negative) height and the radius of the circle.
    Outputs:
        spheric_poses: (n_poses, 3, 4) the poses in the circular path
    """

    def spheric_pose(theta, phi, radius):
        trans_t = lambda t: np.array([
            [1, 0, 0, 0],
            [0, 1, 0, -0.9 * t],
            [0, 0, 1, t],
            [0, 0, 0, 1],
        ])

        rot_phi = lambda phi: np.array([
            [1, 0, 0, 0],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1],
        ])

        rot_theta = lambda th: np.array([
            [np.cos(th), 0, -np.sin(th), 0],
            [0, 1, 0, 0],
            [np.sin(th), 0, np.cos(th), 0],
            [0, 0, 0, 1],
        ])

        c2w = rot_theta(theta) @ rot_phi(phi) @ trans_t(radius)
        c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
        return c2w[:3]

    spheric_poses = []
    for th in np.linspace(0, 2 * np.pi, n_poses + 1)[:-1]:
        spheric_poses += [spheric_pose(th, -np.pi / 5, radius)]  # 36 degree view downwards
    return np.stack(spheric_poses, 0)

def nerf_video_path(c2ws, theta_range=10,phi_range=20,N_views=120):
    c2ws = torch.tensor(c2ws)
    mean_position = torch.mean(c2ws[:,:3, 3],dim=0).reshape(1,3).cpu().numpy()
    rotvec = []
    for i in range(c2ws.shape[0]):
        r = R.from_matrix(c2ws[i, :3, :3])
        euler_ange = r.as_euler('xyz', degrees=True).reshape(1, 3)
        if i:
            mask = np.abs(euler_ange - rotvec[0])>180
            euler_ange[mask] += 360.0
        rotvec.append(euler_ange)
    # 采用欧拉角做平均的方法求旋转矩阵的平均
    rotvec = np.mean(np.stack(rotvec), axis=0)
#     render_poses = [pose_spherical_nerf(rotvec)]
    render_poses = [pose_spherical_nerf(rotvec+np.array([angle,0.0,-phi_range])) for angle in np.linspace(-theta_range,theta_range,N_views//4, endpoint=False)]
    render_poses += [pose_spherical_nerf(rotvec+np.array([theta_range,0.0,angle])) for angle in np.linspace(-phi_range,phi_range,N_views//4, endpoint=False)]
    render_poses += [pose_spherical_nerf(rotvec+np.array([angle,0.0,phi_range])) for angle in np.linspace(theta_range,-theta_range,N_views//4, endpoint=False)]
    render_poses += [pose_spherical_nerf(rotvec+np.array([-theta_range,0.0,angle])) for angle in np.linspace(phi_range,-phi_range,N_views//4, endpoint=False)]
    # render_poses = torch.from_numpy(np.stack(render_poses)).float().to(device)
    return render_poses
