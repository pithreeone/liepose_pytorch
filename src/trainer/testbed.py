import torch
from theseus.geometry import SO3
from torchvision import transforms
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

# From our customed package
from ..data.symsol import dataset
from ..dist import LieDist
from ..metrics import so3 as lie_metrics
from ..model import Model
from ..noise import PowerNoiseSchedule
from ..utils import ops
from ..visualizer import *

import matplotlib.pyplot as plt
import numpy as np
import torch.jit
import csv
from tqdm import tqdm
import datetime
import os
import cv2 as cv
import time

# from torch.multiprocessing import Pool, cpu_count, set_start_method
import torch.multiprocessing as mp

# Set print options to show all elements
torch.set_printoptions(threshold=torch.inf)

def process_head(head, time_seq, features, rt_chunks):
    process_id = os.getpid()
    # print(process_id, len(rt_chunks))
    start = time.time()
    with torch.no_grad():
        chunk_size = len(rt_chunks)
        for t, tp in time_seq:
            tt_chunks = torch.tensor(np.full([chunk_size, 1], t, dtype = np.int32))

            mu = head(features, rt_chunks, tt_chunks)
            # rt_chunks = p_sample_apply(mu, rt_chunks, t) #size(batch_size*n_slices, 3)

    end = time.time()
    print(end - start)

    return rt_chunks

class Testbed():
    def __init__(self, config):
        """
        Args:
        config (SimpleNamespace): Configuration object containing hyperparameters and settings.
        """
        self.a = config
        
        # Create a noise schedule object for sampling noise during training
        # TODO: Try different noise scheduler.
        self.noise_schedule = PowerNoiseSchedule(
            alpha_start=self.a.noise_start, 
            alpha_end=self.a.noise_end,
            timesteps=self.a.timesteps,
            power=self.a.power,
        )
        
        # get representation size based on the chosen representation
        repr_size = lie_metrics.get_repr_size(self.a.repr_type)
        size = self.a.img_res

        # Initialize the model
        self.model = Model(in_dim = repr_size,
                           out_dim = repr_size,
                           image_shape = [1, 3, size, size],
                           resnet_depth = self.a.resnet_depth,
                           mlp_layers = self.a.mlp_layers,
                           fourier_block = self.a.fourier_block,
                           activ_fn = self.a.activ_fn
                           )
        
        # Initialize the EMA-model
        decay = self.a.ema_tau
        self.ema_model = AveragedModel(self.model, multi_avg_fn=get_ema_multi_avg_fn(decay))

        seed_value = 42
        torch.manual_seed(seed_value)
        np.random.seed(seed_value)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1))
        ])
    
    # --- inference ---
    def get_flat_batch_test(self, img, n_slices):
        # torch.manual_seed(42)
        batch = {}
        batch_size = img.shape[0]
        rts = []
        for _ in range(n_slices):
            rt = LieDist._sample_unit(n=(batch_size,))
            rt = lie_metrics.as_repr(rt, self.a.repr_type)
            rts.append(rt)

        batch["img"] = img
        batch["rt"] = torch.cat(rts, dim = 0)  #size(batch*n_slice, 3)
        return batch
    
    def showImage(self, img):
        # [-0.5, 0.5] -> [0, 1]
        img = img + 0.5

        # Convert image color, RGB->BGR
        img_bgr = cv.cvtColor(np.transpose(img[0].cpu().numpy(), (1, 2, 0)), cv.COLOR_RGB2BGR)
        cv.imshow("image", img_bgr)

    def randomWalkSampling(self, head, features, n_slices=1):
        device = next(head.parameters()).device
        # steps = self.a.steps
        steps = 100
        batch_size = features.shape[0]

        time_arr = np.linspace(self.noise_schedule.timesteps-1, 0, int(steps))
        poses = lie_metrics.as_mat(LieDist._sample_unit(n=(batch_size * n_slices,)).to(device))

        for t in time_arr:
            tt = torch.tensor(np.full([batch_size * n_slices, 1], t, dtype = np.int32)).to(device)
            # start_time = time.time()
            mu = head(features, lie_metrics.as_repr(poses, self.a.repr_type), tt)
            # end_time = time.time()
            # total_head_time += end_time - start_time

            # start_time = time.time()
            # rt = self.p_sample_apply(mu, rt, t) #size(batch_size*n_slices, 3)

            size = mu.shape[0] # batch_size*n_slices
            t = np.full([size], t, dtype = np.int32)

            sigma_t = self.noise_schedule.sqrt_alphas[t]
            sigma_L = np.full([size], (self.a.noise_start) ** 0.5, dtype = np.float32)  
            sigma_t = torch.tensor(sigma_t).unsqueeze(dim = 1)  #size(batch_size*n_slices, 1)
            sigma_L = torch.tensor(sigma_L).unsqueeze(dim = 1)  #size(batch_size*n_slices, 1)            

            epsilon = 2e-8
            step_size = (epsilon * 0.5 * (sigma_t ** 2) / (sigma_L ** 2)).to(device)  #size(batch_size*n_slices, 1)
            noise = (LieDist._sample_unit(n=(size,))).to(device) #size(batch_size*n_slices, 3)
            poses = torch.bmm(poses, lie_metrics.as_mat(step_size * mu / sigma_t.to(device) + 0.01 * torch.sqrt(2 * step_size) * noise))  #size(batch_size*n_slices, 3, 3)

            # end_time = time.time()
            # total_sample_time += end_time - start_time

        poses = poses.cpu()
        # print(f"Total head time: {total_head_time}, Total sample time: {total_sample_time}")

        return poses

    def picardIterationSampling(self, head, features, n_slices=1):
        device = next(head.parameters()).device
        size = features.shape[0] * n_slices
        T = self.a.steps
        T_split  = self.a.picard.T_split
        time_arr = torch.linspace(0, T-1, T_split).long()
        time_n_slices = time_arr.repeat_interleave(size).flip(0).to(device)

        if self.a.picard.real_drift == True:
            coeff = self.noise_schedule.get_gradient(T).to(device)
        else:
            coeff = torch.tensor(self.noise_schedule.sqrt_alphas[time_arr], device=device).flip(0)

        coeff_n_slices = coeff.repeat_interleave(size).unsqueeze(-1)
        sample_unit = lie_metrics.as_mat(LieDist._sample_unit(n = (1 * size,))).to(device)
        poses = sample_unit.repeat(T_split+1, 1, 1)

        with torch.no_grad():
            # Perform the loop over K iterations
            for k in range(self.a.picard.iteration):
                # Initialize prefix-mul and s (result of head)
                prefix_mul = torch.eye(3).unsqueeze(0).repeat((T_split + 1) * size, 1, 1).to(device) # (T+1, 3, 3)

                # parallelized calculate s(x_t, t)
                mu = head(features, lie_metrics.as_tan(poses[:(T_split * size)]).to(device), time_n_slices)

                # calculate f(x, t) for SDE
                s = lie_metrics.as_mat(self.a.picard.epsilon * mu * coeff_n_slices).to(device)
                for t in range(T_split):
                    index1 = t * size
                    index2 = (t + 1) * size
                    prefix_mul[index2:index2+size] = torch.bmm(prefix_mul[index1:index1+size], s[index1:index1+size])

                # print(prefix_mul)

                # batch matrix multiplication
                poses = torch.bmm(poses[:size].repeat(T_split + 1, 1, 1), prefix_mul)

        return poses[(T_split * size):].cpu()

    def picardIterationSamplingWindow(self, head, features, n_slices=1):
        device = next(head.parameters()).device
        size = features.shape[0] * n_slices
        T, T_split = self.a.steps, self.a.picard.T_split
        time_arr = torch.linspace(0, T-1, T_split).long().flip(0)

        # expand time_arr & sqrt_qlphas in batch (multiple image or multiple sample/image)

        if self.a.picard.real_drift == True:
            coeff = self.noise_schedule.get_gradient(T).to(device).flip(0)[time_arr]
            # print(coeff)
        else:
            coeff = torch.tensor(self.noise_schedule.sqrt_alphas[time_arr], device=device)
        
        # print(coeff)
        time_n_slices = time_arr.repeat_interleave(size).to(device)
        coeff_n_slices = coeff.repeat_interleave(size).unsqueeze(-1)

        # Parameters for picard iteration
        t = 0
        p = self.a.picard.sliding_window # Batch window size
        threshold = self.a.picard.threshold
        tau = self.a.picard.threshold_tau

        # Initialize poses
        poses = torch.zeros((T_split+1)*size, 3, 3).to(device)

        # Sample initial condition from prior
        poses[:1*size] = lie_metrics.as_mat(LieDist._sample_unit(n = (1 * size,))).to(device)
        poses[1*size:(p+1)*size] = poses[:1*size].repeat(p, 1, 1).to(device)
        
        iterate_index = 0 # Calculate number of iteration for picard iteration

        with torch.no_grad():
            # poses[t] is okay for this iteration, update poses[t+1~t+p]
            while t < T_split:
                # Initialize prefix-mul
                prefix_mul = torch.eye(3).unsqueeze(0).repeat((p+1) * size, 1, 1).to(device) # (T+1, 3, 3)

                # Calculate s(x_t, t) in parallel
                mu = head(features, lie_metrics.as_tan(poses[t*size:(t+p)*size]).to(device), time_n_slices[t*size:(t+p)*size])

                # Calculate drifts for SDE
                s = lie_metrics.as_mat((self.a.picard.epsilon * mu * coeff_n_slices[t*size:(t+p)*size])).to(device)
                for j in range(p):
                    index1, index2 = (j) * size, (j+1) * size
                    prefix_mul[index2:index2+size] = torch.bmm(prefix_mul[index1:index1+size], s[index1:index1+size])
                    # print(prefix_mul)

                # Discretized Picard iteration (Batch matrix multiplication)
                poses_new = torch.bmm(poses[t*size:(t+1)*size].repeat(p+1, 1, 1), prefix_mul)

                # Calculate error for each timestep
                batch_trace = torch.bmm(torch.linalg.inv(poses_new[1*size:(p)*size]), poses[(t+1)*size:(t+p)*size]).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
                error = torch.rad2deg(torch.acos(torch.clamp((batch_trace - 1) / 2, min=-1.0, max=1.0)))
                # threshold = coeff[t] * tau

                valid_indices = (error > threshold).nonzero(as_tuple=True)[0]

                # stride ← min {j : error[j] > τ 2σ^2[j]} ∪ {p} 
                stride = (valid_indices.min().item() // size) + 1 if valid_indices.numel() > 0 else p

                # Updata k+1 poses
                poses[(t+1)*size:(t+p+1)*size] = poses_new[1*size:]

                # Initialize new points that the window now covers
                if t + p + stride + 1 > T_split + 1:
                    remaining = (T_split+1) - (t+p+1)
                    poses[(t+p+1)*size:] = poses[(t+p)*size:(t+p+1)*size].repeat(remaining, 1, 1)
                else:
                    poses[(t+p+1)*size:(t+p+stride+1)*size] = poses[(t+p)*size:(t+p+1)*size].repeat(stride, 1, 1)

                # if t == 99:

                t += stride
                p = min(p, T_split - t)

                iterate_index += 1
                # time.sleep(0.2)


        return poses[(T_split * size):].cpu(), iterate_index

    def test(self):
        transform = self.transform
        batch_size = 1

        test_dataset = dataset.load_symmetric_solids_dataset(split='test', transform=transform)
        test_loader = dataset.getDataLoader(test_dataset, batch_size = batch_size, shuffle=False, num_workers=0)

        # Check if CUDA is available and set the device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[INFO] Using device: {device}")

        # Load weights
        # model = torch.load(".log/model_250000.pth", weights_only=False)
        model = torch.load(".log/2025-06-10_17-07-26/model_100000.pth", weights_only=False)
        model.eval()

        backbone, head = model.backbone, model.head
        backbone, head = backbone.to(device), head.to(device)

        n_slices = 1
        average_minimum_angle_no_mark = 0
        average_minimum_angle = 0
        considered_shape = ["Tetrahedron", "Cube", "Icosahedron", "Cone", "Cylinder", "Marked Tetrahedron", "Marked Cube", "Marked Icosahedron"]
        shape_count = [0 for _ in range(8)]
        average_minimum_angle_each_shape = [0 for _ in range(8)]
        total_no_mark = 0
        total = 0

        iteration = 0
        avg_iteration = 0

        print(f"[INFO] Preparing for evaluation with {len(test_loader)} test batches.")
        print("[INFO] Starting model evaluation...")
        start_sample = time.time()

        for batch_idx, (img, rotation, rotations_equivalent) in tqdm(enumerate(test_loader), total=len(test_loader)):
            label_shapes = []
            for idx in range(batch_size*batch_idx, batch_size*batch_idx+len(img)):
                label_shapes.append(test_dataset.get_label(idx))
                shape_count[test_dataset.get_label(idx)] += 1

            # Step 1: Backbone: get features of image via backbone network
            img = img.to(device)

            start_backbone = time.time()
            features = backbone(img)
            end_backbone = time.time()
            # print(f"Time backbone: {end_backbone - start_backbone:.6f} seconds")

            # Step 2: Denoised pose (sampling)
            start_sampling = time.time()
            # poses = self.randomWalkSampling(head, features, n_slices)
            # poses = self.picardIterationSampling(head, features, n_slices)
            poses, iteration = self.picardIterationSamplingWindow(head, features, n_slices)
            avg_iteration += iteration
            end_sampling = time.time()
            # print(f"Sampling Time: {end_sampling - start_sampling:.6f} seconds, ")

            # Step 3: Evaluation - Calculate minimun angle
            rt_idx = 0 # rt_idx is the index of rt, size [batch_size*n_slices, 3]
            for sample_idx in range(len(img)):
                # get ground-truth rotations of current sample
                rotations = rotations_equivalent[sample_idx].cpu().numpy()

                # get predicted rotations of current sample
                predict_r = poses[sample_idx]

                # Find the minimum angle from those equivalent answers
                min_angle = 1000000
                min_rotations_idx = -1
                # print(f"Find the minimum angle of totally {len(rotations)} possible solutions.")
                angles = []

                for rot_idx, rotation in enumerate(rotations):
                    # Compute the relative rotation matrix & trace
                    trace_R_rel = np.clip(np.trace(np.dot(predict_r.T.detach().numpy(), rotation)), -1, 3)

                    # Compute the angular distance (in radians)
                    angle = np.degrees(np.arccos((trace_R_rel - 1) / 2))
                    angles.append(angle)
                    
                    if angle < min_angle:
                        min_angle = angle
                        min_rotations_idx = rot_idx

                average_minimum_angle_each_shape[label_shapes[sample_idx]] += min_angle
                average_minimum_angle += min_angle
                total += 1

                if len(rotations) != 1:
                    average_minimum_angle_no_mark += min_angle
                    total_no_mark += 1
                    # tqdm.write(f"{min_angle}")
                    # if total % 200 == 0:
                    #     end_sample = time.time()
                    #     avg_sample_time = (end_sample - start_sample) / (batch_idx + 1)
                        # tqdm.write(f"Total {total} samples, average minimum angle: {average_minimum_angle / total:.5f},"
                        #             f"average time: {avg_sample_time:.5f}, average iteration: {avg_iteration / (batch_idx+1):.5f}")
                        
                        # tqdm.write(f"[INFO] Total {total:>5} samples | Average Minimum Angular Distance: {average_minimum_angle / total:.5f} | Average Time: {avg_sample_time:.5f}")

                # Update index of rt
                rt_idx += 1

                # print(f"Minimum angle: {min_angle}, \npredict: \n{predict_r}, \nmin-corresponding answer: \n{rotations[min_rotations_idx]}")
                # break

        end_sample = time.time()
        avg_sample_time = (end_sample - start_sample) / len(test_loader)
            

            # self.showImage(img)

            # Wait for user input
            # key = cv.waitKey(0)
            # if key == ord('q'):
            #     print("Exiting...")
            #     break
            # elif key == ord(' '):
            #     print("Next image...")
            #     cv.destroyAllWindows()

        average_minimum_angle = average_minimum_angle / total
        average_minimum_angle_no_mark = average_minimum_angle_no_mark / total_no_mark
        print("-" * 130)
        print(f"{'Summary of Evaluation Results':^130}")
        print("-" * 130)
        print(f"{'Overall':<20} | Total {total:>6} samples | Average Minimum Angular Distance: {average_minimum_angle:>10.6f} | Average Time: {avg_sample_time:.5f}")
        print(f"{'Unmarked Only':<20} | Total {total_no_mark:>6} samples | Average Minimum Angular Distance: {average_minimum_angle_no_mark:>10.6f}")
        print("-" * 130)

        print(f"{'Per Shape Statistics':^130}")  # <-- Informative header centered
        print("-" * 130)

        for i in range(len(considered_shape)):
            average_minimum_angle_each_shape[i] = average_minimum_angle_each_shape[i]/shape_count[i]
            print(f"{considered_shape[i]:<20} | Total {shape_count[i]:>6} samples | Average Minimum Angular Distance: {average_minimum_angle_each_shape[i]:>10.6f}")
        print("-" * 130)

    def visualize(self):
        transform = self.transform

        batch_size = 1
        test_dataset = dataset.load_symmetric_solids_dataset(split='test', transform=transform)
        test_loader = dataset.getDataLoader(test_dataset, batch_size = batch_size, shuffle=True, num_workers=10)

        # Check if CUDA is available and set the device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load weights
        model = torch.load(".log/model_250000.pth", weights_only=False)
        model.eval()

        backbone, head = model.backbone, model.head
        backbone, head = backbone.to(device), head.to(device)

        # # figure initialization
        fig, axs = init_fig()

        n_slices = 2000

        with torch.no_grad():
            for batch_idx, (img, _, rotations_equivalent) in enumerate(test_loader):
                # Step 1: Pre-processing
                img = img.to(device)

                # Step 2: Backbone: get features of image via backbone network
                features = backbone(img)

                # Step 3: Denoised pose (sampling)
                poses, _ = self.picardIterationSamplingWindow(head, features, n_slices)

                img_frame = img[0].cpu()
                img_np = img_frame.permute(1, 2, 0).numpy()
                img_rgb = cv.cvtColor(img_np, cv.COLOR_BGR2RGB)
                img_rgb += 0.5
                show_frame(img_rgb, axs[0])

                visualize_so3_probabilities(poses, fig=fig, ax=axs[1])

                fig.savefig(f"vis/test_data/frame{batch_idx}.png", bbox_inches='tight', pad_inches=0.1)

                # if cv.waitKey(30) & 0xFF == ord('q'):
                #     print("Exiting...")
                #     break

    def visualize_video(self):
        transform = self.transform

        cap = cv.VideoCapture("vis/cylinder.mp4")
        if not cap.isOpened():
            print("Error: Cannot open video.")
            return

        # Check if CUDA is available and set the device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load weights
        model = torch.load(".log/model_250000.pth", weights_only=False)
        model.eval()

        backbone, head = model.backbone, model.head
        backbone, head = backbone.to(device), head.to(device)
        
        # # figure initialization
        fig, axs = init_fig()

        # set_start_method('spawn', force=True)  # Needed for multiprocessing
        n_slices = 2000
        frame_id = 0

        avg_loop_time = 0
        avg_sample_time = 0

        with torch.no_grad():
            while cap.isOpened():
                start_frame = time.time()

                # Read a frame from the video
                ret, frame  = cap.read()
                if not ret:
                    print("End of video.")
                    break
                
                # Step 1: Pre-processing
                img_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                img = transform(img_rgb).unsqueeze(0).to(device)

                end_preprocess = time.time()
                print(f"----------------------------- {frame_id} -------------------------------")
                print(f"Data Preprocessing time: {end_preprocess - start_frame:.6f} seconds")

                # Step 2: Backbone: get features of image via backbone network
                start_backbone = time.time()
                features = backbone(img)
                end_backbone = time.time()
                print(f"Time backbone: {end_backbone - start_backbone:.6f} seconds")

                # Step 3: Denoised pose (sampling)
                start_sampling = time.time()

                # poses = self.picardIterationSampling(head, features, n_slices)
                poses, _ = self.picardIterationSamplingWindow(head, features, n_slices)
                # poses = self.randomWalkSampling(head, features, n_slices)

                end_sampling = time.time()
                print(f"Sampling Time: {end_sampling - start_sampling:.6f} seconds, ")

                # print(f"Time_seq Loop Time: {end_loop - start_loop:.6f} seconds, "
                #     f"(Time head: {head_time:.6f} seconds, Time sample: {sample_time:.6f} seconds)")

                # Step 4: Visualization, draw image
                start_draw = time.time()

                show_frame(frame, axs[0])

                visualize_so3_probabilities(poses, fig=fig, ax=axs[1])

                fig.savefig(f"vis/video_result/frame{frame_id}.png", bbox_inches='tight', pad_inches=0.1)

                frame_id += 1

                end_draw = time.time()

                print(f"Draw Time: {end_draw - start_draw:.6f} seconds")

                # if cv.waitKey(30) & 0xFF == ord('q'):
                #     print("Exiting...")
                #     break
                
                end_frame = time.time()
  
                print(f"One Frame Inference Time: {end_frame - start_frame:.6f} seconds")
                avg_sample_time += end_frame-start_frame

                if frame_id % 20 == 0:
                    print(f"Avg Inference Time: {avg_sample_time / frame_id}, Avg Loop Time: {avg_loop_time / frame_id}")
    
            # Release video capture and close OpenCV windows
            cap.release()
            cv.destroyAllWindows()
            print(f"Avg Inference Time: {avg_sample_time / frame_id}, Avg Loop Time: {avg_loop_time / frame_id}")

    # --- train ---
    def get_flat_batch_train(self, img, rot, n_slices):
        """
        Diffusing each image.

        Args: 
        img (torch.Tensor): Input images of shape (batch_size, channels, height, width)
        rot (torch.Tensor): Rotation matrices (label) of shape (batch_size, 3, 3)
        n_slices (int): Number of noisy samples per image.

        Returns:
        dict: A dictionary containing augmented data for training
            - "img" (torch.Tensor): Images, size (batch_size, channels, height, width), e.g., (16, 3, 224, 224)
            - "rt" (torch.Tensor): Noisy rotations, concatenated across slices, size (batch_size * n_slices, 3), e.g., [2048, 3]
            - "t" (torch.Tensor): Noise schedule timesteps, size (batch_size * n_slices), e.g., (2048)
            - "zt" (torch.Tensor): Noise samples in Lie algebra space, size (batch_size * n_slices, 3), e.g., (2048, 3)
            - "r0" (torch.Tensor): Ground truth rotations in Lie algebra space, size (batch_size * n_slices, 3), e.g., (2048, 3)
        """
        batch = {}
        rts, ts, zts, r0s, tas = [], [], [], [], []

        batch_size = img.shape[0]
        for _ in range(n_slices):
            # Sample random timesteps for size = batch_size
            t = torch.randint(low=0, high=self.noise_schedule.timesteps, size=(batch_size,)) #size(batch,)
            # t = torch.tensor([50] * batch_size)

            # Convert rotations to SO(3) representation, type: SO3, size: (batch, 3, 3)
            r0 = lie_metrics.as_lie(rot) 
            
            # Sample unit Gaussian noise, type: tensor, size: (batch, 3)
            zt = LieDist._sample_unit(n=(batch_size,))

            # alphat = torch.tensor(self.noise_schedule.alphas[t]).unsqueeze(1)
            sqrt_alphas_t = torch.tensor(self.noise_schedule.sqrt_alphas[t]).unsqueeze(1)

            # Target of model, formula(9) from paper in 2024 CVPR (Confronting Ambiguity...)
            ta = -zt
            # ta = -1 / sqrt_alphas_t * zt

            # Add noise to rotations, type: SO3, size: (batch, 3, 3)
            rt = ops.add(r0, SO3.exp_map(sqrt_alphas_t * zt))

            # Convert rotation_0, rotation_t, noise_t into the specified representation
            r0 = lie_metrics.as_repr(r0, self.a.repr_type) # size(batch, 3)
            zt = lie_metrics.as_repr(zt, self.a.repr_type) # size(batch, 3)
            rt = lie_metrics.as_repr(rt, self.a.repr_type) # size(batch, 3)
            ta = lie_metrics.as_repr(ta, self.a.repr_type)

            # Collect the slices
            rts.append(rt)
            ts.append(t)
            zts.append(zt)
            r0s.append(r0)
            tas.append(ta)

        # Concatenate slices and add them to the batch, e.g. [img1-1, img1-2, img1-3, ..., img2-1, img2-2, ...]
        batch["img"] = img
        batch["rt"] = torch.cat(rts, dim=0)
        batch["t"] = torch.cat(ts, dim=0)
        batch["zt"] = torch.cat(zts, dim=0)
        batch["r0"] = torch.cat(r0s, dim=0)
        batch["ta"] = torch.cat(tas, dim=0)

        return batch

    def train(self):
        """
        Main training loop for the model
        """
        # Define data transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1))
        ])

        # Load datasets for training and testing
        train_dataset = dataset.load_symmetric_solids_dataset(split='train', transform=transform)
        train_loader = dataset.getDataLoader(train_dataset, batch_size = self.a.batch_size, shuffle=False, num_workers=10)
        
        # Check if CUDA is available and set the device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Info] Using device: {device}")

        # Prepare the model
        # model = torch.jit.script(self.model)
        model = self.model.to(device)
        model.train()  # Set the model to training mode

        
        # Define optimizer and learning rate scheduler
        optim = torch.optim.AdamW(model.parameters(), lr=self.a.init_lr)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=self.a.lr_decay_rate)
        
        # Training data
        train_data_iter = iter(train_loader)
        img, rot, _ = next(train_data_iter)
        batch_idx, epoch_idx = 0, 0

        # Record data and write to csv file
        record_data = []
        avg_loss = 0

        # Get the current date and time
        current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        # current_time = '000'

        # Create folder
        filename = "training.csv"
        dir_path = os.path.join(".log", current_time)

        file_path = os.path.join(dir_path, filename)
        os.makedirs(dir_path, exist_ok=True)

        torch.manual_seed(42)

        with tqdm(total=self.a.train_steps, desc="Train loss: ", dynamic_ncols=True) as pbar:
            # Training steps (update parameters)
            for step in range(self.a.train_steps):
                # Prepare the training batch
                batch = self.get_flat_batch_train(img, rot, self.a.n_slices)
                img, rt, t, ta = batch["img"], batch["rt"], batch["t"].reshape((-1, 1)), batch['ta']
                img, rt, t, ta = img.to(device), rt.to(device), t.to(device), ta.to(device)

                # Forward pass: predict score values
                mu = model(img, rt, t)

                # Compute loss
                loss = torch.mean(lie_metrics.distance_fn(ta, mu, self.a.loss_name))
                avg_loss += loss.item()

                # Backpropagation and optimization
                optim.zero_grad()
                loss.backward()
                optim.step()
                
                # Update ema-model parameters every s steps, [0, s, 2s, ...]
                if batch_idx % self.a.ema_steps == 0:
                    self.ema_model.update_parameters(model)

                # Get the next batch of image and gt-rotation matrix from train_data_iter
                try:
                    img, rot, _ = next(train_data_iter)
                except StopIteration:
                    # Finish iterating all the training data. Next epoch
                    train_data_iter = iter(train_loader)
                    img, rot, _ = next(train_data_iter)

                    # Reset batch index and increase epoch index
                    batch_idx = 0
                    epoch_idx += 1

                    # Update learning rate
                    lr_scheduler.step()
                    for param_group in optim.param_groups:
                        param_group['lr'] = max(param_group['lr'], self.a.end_lr)
                    
                    # Print current learning rate
                    current_lr = optim.param_groups[0]['lr']
                    print(f"Epoch {epoch_idx + 1}: Updated Learning Rate = {current_lr:.6f}")

                # Update tqdm description and progress
                pbar.set_description(f"Train loss: {loss:.4f}")
                pbar.update(1)
                
                # Log progress every b batches, [b-1, 2b-1, 3b-1, ...]
                if (batch_idx + 1) % 20 == 0:
                    tqdm.write(f"Batch {batch_idx+1}: Train Loss {avg_loss / 20}")
                    record_data.append([step + 1, epoch_idx, batch_idx + 1, avg_loss / 20])
                    avg_loss = 0
                
                # save check-point model
                if (step + 1) % self.a.ckpt_steps == 0:
                    model_path = os.path.join(dir_path, f"model_{step + 1}.pth")
                    torch.save(model, model_path)
                    print(f"Save check-point model to '{model_path}'")

                # Update batch index and epoch index
                batch_idx += 1
        
        print("Finish training process!")

        # Save the training information to csv file
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['step', 'epoch', 'batch', 'loss'])
            writer.writerows(record_data)
        
        print(f"Training info has been saved to '{file_path}'")

        # Save model
        torch.save(model, os.path.join(dir_path, "model.pth"))
        # torch.jit.save(model, os.path.join(dir_path, "model.pth"))