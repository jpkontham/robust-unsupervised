import os
import datetime
import tqdm
import torch
import glob
from torchvision.utils import save_image

# Repo-specific imports
from cli import parse_config
import benchmark
from benchmark import Task, Degradation
from robust_unsupervised import *

# 1. Load configuration
config = parse_config()
benchmark.config.resolution = config.resolution

print(f"Project: {config.name}")
timestamp = datetime.datetime.now().isoformat(timespec="seconds").replace(":", "")

# 2. Prepare Models & Loss (Force to GPU)
G, D = open_models(config.pkl_path)
G = G.cuda().eval()
if D is not None:
    D = D.cuda().eval()

loss_fn = MultiscaleLPIPS().cuda()

# 3. Fixed run_phase (now accepts target and degradation)
def run_phase(label: str, variable: Variable, lr: float, target: torch.Tensor, degradation: Any):        
    optimizer = NGD(variable.parameters(), lr=lr)
    try:
        # Optimization loop (150 iterations)
        for _ in tqdm.tqdm(range(150), desc=label, leave=False):
            x = variable.to_image()
            # Calculate loss: comparing degraded prediction vs damaged target
            loss = loss_fn(degradation.degrade_prediction, x, target, degradation.mask).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    except KeyboardInterrupt:
        pass

    # Save results for this phase
    suffix = "_" + label
    pred = resize_for_logging(variable.to_image(), config.resolution)
    save_image(pred, f"pred{suffix}.png", padding=0)

if __name__ == '__main__':
    # Task selection logic
    if config.tasks == "single":
        tasks = benchmark.single_tasks
    elif config.tasks == "composed":
        tasks = benchmark.composed_tasks
    elif config.tasks == "all":
        tasks = benchmark.all_tasks
    else:
        raise Exception("Invalid task name")
    
    for task in tasks:
        experiment_path = f"out/{config.name}/{timestamp}/{task.category}/{task.name}/{task.level}/"
        
        image_paths = sorted(
            glob.glob(config.dataset_path + "/**/*.png", recursive=True) +
            glob.glob(config.dataset_path + "/**/*.jpg", recursive=True)
        )
        assert len(image_paths) > 0, "No images found!"

        with directory(experiment_path):
            for j, image_path in enumerate(image_paths):
                with directory(f"inversions/{j:04d}"):
                    # Load images onto GPU
                    ground_truth = open_image(image_path, config.resolution).cuda()
                    degradation = task.init_degradation()
                    target = degradation.degrade_ground_truth(ground_truth).cuda()
                    
                    save_image(ground_truth, "ground_truth.png")
                    save_image(target, "target.png")
                    
                    # Phase I: W Space
                    W_variable = WVariable.sample_from(G)
                    run_phase("W", W_variable, config.global_lr_scale * 0.08, target, degradation)

                    # Phase II: W+ Space
                    Wp_variable = WpVariable.from_W(W_variable)
                    run_phase("W+", Wp_variable, config.global_lr_scale * 0.02, target, degradation)

                    # Phase III: S-Space (StyleSpace) with Hook
                    S_variable = SVariable.from_Wp(Wp_variable)
                    hook = StyleGAN3Hook(G, S_variable.to_input_tensor())
                    try:
                        run_phase("S", S_variable, config.global_lr_scale * 0.005, target, degradation)
                    finally:
                        hook.remove() # Crucial: prevent memory leaks
