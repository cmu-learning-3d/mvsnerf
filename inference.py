import sys,os,imageio

from render_utils import *

# root = '/home/hengfei/Desktop/research/mvsnerf'
root = '/mnt/aidtr/members/vanshajc/mvsnerf'
os.chdir(root)
sys.path.append(root)

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

torch.cuda.set_device(0)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

for i_scene, scene in enumerate([1]):# any scene index, like 1,2,3...,,8,21,103,114

    cmd = f'--datadir /tmp/mvs_training/dtu/scan{scene} \
     --dataset_name dtu_ft --imgScale_test {1.0} ' #--use_color_volume
    
    is_finetued = False # set False if rendering without finetuning
    if is_finetued:
        cmd += f'--ckpt ./runs_fine_tuning/dtu_scan{scene}_2h/ckpts//latest.tar'
    else:
        cmd += '--ckpt ./ckpts/mvsnerf-v0.tar'

    args = config_parser(cmd.split())
    args.use_viewdirs = True

    args.N_samples = 128
    args.feat_dim =  8+3*4
    args.N_importance = 64
    args.use_color_volume = False if not is_finetued else args.use_color_volume

    # create models
    if i_scene==0 or is_finetued:
        render_kwargs_train, render_kwargs_test, start, grad_vars = create_nerf_mvs(args, use_mvs=True, dir_embedder=False, pts_embedder=True)
        filter_keys(render_kwargs_train)

        MVSNet = render_kwargs_train['network_mvs']

        render_kwargs_train.pop('network_mvs')


    datadir = args.datadir
    datatype = 'val'
    pad = 24 #the padding value should be same as your finetuning ckpt
    args.chunk = 5120
    frames = 60


    dataset = dataset_dict[args.dataset_name](args, split=datatype)
    print(dataset)
    val_idx = dataset.img_idx
    
    save_as_image = False
    save_dir = f'results/video2' 
    os.makedirs(save_dir, exist_ok=True)
    MVSNet.train()
    MVSNet = MVSNet.cuda()
    
    with torch.no_grad():
        
        if is_finetued:   
            # large baselien
            imgs_source, proj_mats, near_far_source, pose_source = dataset.read_source_views(device=device)
            volume_feature = torch.load(args.ckpt)['volume']['feat_volume']
            volume_feature = RefVolume(volume_feature.detach()).cuda()
        else:            
            # neighboring views with angle distance
            c2ws_all = dataset.load_poses_all()
            random_selete = torch.randint(0,len(c2ws_all),(1,)) #!!!!!!!!!! you may change this line if rendering a specify view 
            random_selete = 0
            dis = np.sum(c2ws_all[:,:3,2] * c2ws_all[[random_selete],:3,2], axis=-1)
            pair_idx = np.argsort(dis)[::-1][:3]#[25, 21, 33]#[14,15,24]#
            imgs_source, proj_mats, near_far_source, pose_source = dataset.read_source_views(pair_idx=pair_idx, device=device)
            volume_feature, _, _ = MVSNet(imgs_source, proj_mats, near_far_source, pad=pad)
            
        imgs_source = unpreprocess(imgs_source)

        c2ws_render = gen_render_path(pose_source['c2ws'].cpu().numpy(), N_views=frames)
        c2ws_render = torch.from_numpy(np.stack(c2ws_render)).float().to(device)
        
        
        try:
            tqdm._instances.clear() 
        except Exception:     
            pass
        
        frames = []
        img_directions = dataset.directions.to(device)
        for i, c2w in enumerate(tqdm(c2ws_render)):
            torch.cuda.empty_cache()
            
            rays_o, rays_d = get_rays(img_directions, c2w)  # both (h*w, 3)
            rays = torch.cat([rays_o, rays_d,
                     near_far_source[0] * torch.ones_like(rays_o[:, :1]),
                     near_far_source[1] * torch.ones_like(rays_o[:, :1])],
                    1).to(device)  # (H*W, 3)
            
        
            N_rays_all = rays.shape[0]
            rgb_rays, depth_rays_preds = [],[]
            for chunk_idx in range(N_rays_all//args.chunk + int(N_rays_all%args.chunk>0)):

                xyz_coarse_sampled, rays_o, rays_d, z_vals = ray_marcher(rays[chunk_idx*args.chunk:(chunk_idx+1)*args.chunk],
                                                    N_samples=args.N_samples)

                # Converting world coordinate to ndc coordinate
                H, W = imgs_source.shape[-2:]
                inv_scale = torch.tensor([W - 1, H - 1]).to(device)
                w2c_ref, intrinsic_ref = pose_source['w2cs'][0], pose_source['intrinsics'][0].clone()
                xyz_NDC = get_ndc_coordinate(w2c_ref, intrinsic_ref, xyz_coarse_sampled, inv_scale,
                                             near=near_far_source[0], far=near_far_source[1], pad=pad*args.imgScale_test)


                # rendering
                rgb, disp, acc, depth_pred, alpha, extras = rendering(args, pose_source, xyz_coarse_sampled,
                                                                       xyz_NDC, z_vals, rays_o, rays_d,
                                                                       volume_feature,imgs_source, **render_kwargs_train)
    
                
                rgb, depth_pred = torch.clamp(rgb.cpu(),0,1.0).numpy(), depth_pred.cpu().numpy()
                rgb_rays.append(rgb)
                depth_rays_preds.append(depth_pred)

            
            depth_rays_preds = np.concatenate(depth_rays_preds).reshape(H, W)
            depth_rays_preds, _ = visualize_depth_numpy(depth_rays_preds, near_far_source)
            
            rgb_rays = np.concatenate(rgb_rays).reshape(H, W, 3)
#             H_crop, W_crop = np.array(rgb_rays.shape[:2])//20
#             rgb_rays = rgb_rays[H_crop:-H_crop,W_crop:-W_crop]
#             depth_rays_preds = depth_rays_preds[H_crop:-H_crop,W_crop:-W_crop]
            img_vis = np.concatenate((rgb_rays*255,depth_rays_preds),axis=1)
            frames.append(img_vis.astype('uint8'))
                
    imageio.mimsave(f'{save_dir}/128_64.gif', np.stack(frames), fps=20)
# plt.imshow(rgb_rays)