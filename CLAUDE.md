# CLAUDE.md

This file provides guidance for working with the GVHMR (Global Video-based Human Mesh Recovery) codebase in the context of the Oshieru golf swing training application.

## Project Overview

GVHMR is the video-to-3D processing pipeline that converts uploaded golf swing videos into USDZ 3D models with metrics. It runs as a GPU worker service, typically deployed on Kaggle with T4 x2 GPU configuration.

**Primary entry point:** `mainserver (2).ipynb` - Jupyter notebook serving as the main processing server

**Core function:**
1. Download golf swing video from Redis queue
2. Run GVHMR to extract SMPL pose parameters (frame-by-frame 3D human pose)
3. Process results in Blender: generate USDZ with transparent body, bone proxies, and golf club proxy
4. Calculate golf swing metrics (X-factor, tempo, weight distribution, etc.)
5. Upload USDZ and metrics JSON back to storage

## Architecture

```
Redis Queue (Upstash)
    ↓
worker_loop() fetches job
    ↓
run_gvhmr() - extracts SMPL parameters (hmr4d_results.pt.pkl)
    ↓
processs_video_output() - Blender processing
    ├── Load SMPL FBX model
    ├── Apply per-frame pose/shape
    ├── Setup transparent body material (α=0.25)
    ├── Remove hands, create golf club cylinder
    ├── Create bone proxies (joint spheres + bone sticks)
    └── Export to USDZ (UsdSkel animation)
    ↓
calculate_golf_swing_metrics_from_armature()
    ├── Sample all joint positions per frame
    ├── Calculate COM, weight distribution
    ├── Compute pelvis/chest rotation, sway, lift
    ├── Detect events (setup, top, impact)
    └── Return metrics JSON
    ↓
Upload USDZ + metrics to storage
```

## Environment Setup

### Requirements
- Python 3.10
- Conda (use `condacolab` on Kaggle)
- CUDA GPU (T4 x2 recommended)

### Installation
```python
# Install conda
!pip install -q condacolab
from condacolab import condacolab
condacolab.install()

# Create environment
!conda create -y -n gvhmr python=3.10
!conda activate gvhmr

# Clone and install GVHMR
!git clone 'https://<TOKEN>@github.com/sksum/GVHMR.git'
%cd GVHMR
!source activate gvhmr && grep -v "^chumpy" requirements.txt | pip install -r /dev/stdin
!source activate gvhmr && pip install -e .
```

### Model Checkpoints (HuggingFace)
Required models downloaded via aria2c:
- `SMPL_NEUTRAL.pkl` - Body model
- `SMPLX_NEUTRAL.npz` - Extended body model
- `dpvo.pth` - Dense Pose Video
- `gvhmr_siga24_release.ckpt` - GVHMR weights
- `epoch=10-step=25000.ckpt` - HMR2
- `vitpose-h-multi-coco.pth` - Pose estimation
- `yolov8x.pt` - Object detection

## Key Components

### 1. GVHMR Processing (`tools/demo/demo.py`)

Run GVHMR on video to extract pose parameters:
```bash
python tools/demo/demo.py --video="<video_path>" -s
```

**Output:** `outputs/demo/<job_id>/hmr4d_results.pt.pkl`

The pickle file contains:
- `smpl_params_global['transl']` - Global translation per frame
- `smpl_params_global['global_orient']` - Root orientation
- `smpl_params_global['body_pose']` - Joint rotations (22 joints × 3)

### 2. Blender Processing (`bpy`)

#### SMPL Bone Mapping
```python
PART_MATCH = {
    'root': 'root', 'bone_00': 'Pelvis', 'bone_01': 'L_Hip', 'bone_02': 'R_Hip',
    'bone_03': 'Spine1', 'bone_04': 'L_Knee', 'bone_05': 'R_Knee', 'bone_06': 'Spine2',
    'bone_07': 'L_Ankle', 'bone_08': 'R_Ankle', 'bone_09': 'Spine3', 'bone_10': 'L_Foot',
    'bone_11': 'R_Foot', 'bone_12': 'Neck', 'bone_13': 'L_Collar', 'bone_14': 'R_Collar',
    'bone_15': 'Head', 'bone_16': 'L_Shoulder', 'bone_17': 'R_Shoulder', 'bone_18': 'L_Elbow',
    'bone_19': 'R_Elbow', 'bone_20': 'L_Wrist', 'bone_21': 'R_Wrist',
    'bone_22': 'L_Hand', 'bone_23': 'R_Hand',
}
```

#### Key Functions

**`rodrigues2bshapes(pose)`** - Convert axis-angle rotation vectors to rotation matrices and blendshapes

**`apply_trans_pose_shape(trans, body_pose, arm_ob, obname, frame)`** - Apply translation and pose to armature for a single frame

**`setup_body_transparent_material(mesh_obj, alpha=0.25)`** - Create semi-transparent body material using Principled BSDF

**`create_bone_proxies(armature_obj, num_frames)`** - Create visualization geometry:
- UV spheres at joint heads (PROXY_RADIUS = 0.035m)
- Cylinders connecting parent→child bones (STICK_RADIUS = 0.012m)
- Cyan color (0.2, 0.6, 1.0)
- All keyframed per frame to follow armature

**`remove_hands_and_create_cylinder(armature_obj, mesh_obj)`** - Remove hand vertices, create golf club proxy cylinder

**`apply_shoulder_ik(armature_obj, target_pos, side, frame)`** - Rotate shoulder to bring hands together (namaste pose), keeping elbow locked

**`animate_cylinder(armature_obj, cylinder, num_frames)`** - Orient cylinder 90° from arm in palm direction (like golf club), length = 0.25m

**`export_with_cylinder(...)`** - Export to USD with:
- Body mesh (transparent, α=0.25)
- Armature (as UsdSkeleton + SkelAnimation)
- Bone proxies (opaque cyan)
- Golf club cylinder

### 3. Golf Swing Metrics

**`calculate_golf_swing_metrics_from_armature(armature_obj, fps=60)`** - Main metrics function

**Joint Sampling:**
```python
JOINTS = {
    "pelvis": "m_avg_Pelvis", "root": "m_avg_root",
    "L_hip": "m_avg_L_Hip", "R_hip": "m_avg_R_Hip",
    # ... (22 joints total)
}
```

**Metrics Calculated:**

| Metric | Description |
|--------|-------------|
| Center of Mass (COM) | Weighted average of body segments |
| Weight Distribution | Lead vs trail foot based on COM position |
| Pelvis Rotation | Horizontal angle of hip line |
| Pelvis Sway | Lateral displacement (X-axis) |
| Pelvis Lift | Vertical displacement (Z-axis) |
| Chest Rotation | Horizontal angle of shoulder line |
| X-Factor | Chest rotation - Pelvis rotation |
| Setup Frame | First significant hand movement |
| Top Frame | Minimum X-factor (maximum coil) |
| Impact Frame | Hands closest to setup position after top |
| Backswing Duration | Top frame - Setup frame |
| Downswing Duration | Impact frame - Top frame |
| Tempo Ratio | Backswing / Downswing |

**Output Structure:**
```json
{
  "events": {
    "setup_frame": N,
    "top_frame": M,
    "impact_frame": K,
    "backswing_duration_frames": ...,
    "downswing_duration_frames": ...
  },
  "tempo_ratio": float,
  "per_frame_metrics": {
    "0": {"weight_distribution": {...}, "pelvis": {...}, "chest": {...}, "x_factor_degrees": ...},
    "1": {...},
    ...
  },
  "summary_metrics": {
    "max_x_factor": ...,
    "min_x_factor": ...,
    "x_factor_range": ...,
    "max_pelvis_rotation": ...,
    "max_chest_rotation": ...,
    "peak_com_sway": ...
  }
}
```

### 4. Worker Loop

Multi-GPU processing with Redis queue:

```python
NUM_GPUS = 2  # Kaggle T4 x2

def worker_loop(gpu_id, job_queue):
    while True:
        job = job_queue.get()
        # Download video
        # Run GVHMR with CUDA_VISIBLE_DEVICES=gpu_id
        # Process in Blender
        # Calculate metrics
        # Upload USDZ + JSON
        # Update Redis status
```

**Redis Keys:**
- `lpop/queue:pending` - Pop next job
- `get/job:{job_id}` - Get job details
- `set/job:{job_id}` - Update job status

## File Structure

```
GVHMR/
├── mainserver (2).ipynb          # Main processing server (THIS FILE)
├── hmr4d/                        # Core GVHMR implementation
├── tools/
│   └── demo/
│       └── demo.py              # GVHMR inference script
├── inputs/
│   └── checkpoints/             # Model weights
│       ├── body_models/
│       │   ├── smpl/
│       │   └── smplx/
│       ├── dpvo/
│       ├── gvhmr/
│       ├── hmr2/
│       ├── vitpose/
│       └── yolo/
├── outputs/
│   └── demo/{job_id}/
│       └── hmr4d_results.pt.pkl  # GVHMR output
└── requirements.txt
```

## Running Locally

To test a single video:

```python
# In notebook cell
job_id = "test-job"
video_path = "/path/to/video.mp4"

# Run GVHMR
!python tools/demo/demo.py --video="{video_path}" -s

# Process in Blender
processs_video_output(
    f'outputs/demo/{job_id}/hmr4d_results.pt.pkl',
    job_id
)

# Outputs:
# - /kaggle/working/outputs/{job_id}.usdz
# - Metrics in scene (can be extracted with calculate_golf_swing_metrics_from_armature)
```

## Known Issues & Notes

- **GPU requirement:** GVHMR requires CUDA GPU. T4 x2 on Kaggle is recommended configuration.
- **Memory:** Processing multiple jobs concurrently requires significant GPU memory.
- **Blender version:** Uses bpy 4.2.0 for USD export with ARKit compatibility.
- **Coordinate system:** Y-up coordinate system; HIGH_FROM_FLOOR = 1.5m offset.
- **Hand removal:** Hands are removed from mesh and replaced with cylinder proxy for golf club visualization.
- **Shoulder IK:** Uses simplified IK that only rotates shoulder (elbow locked) to bring hands to midpoint.

## Dependencies

Key Python packages:
- `bpy==4.2.0` - Blender Python API
- `usd-core` - Pixar USD library
- `torch` - PyTorch (GPU)
- `numpy` - Numerical operations
- `requests` - HTTP client for Redis/storage
- `joblib` - For pickle loading

## Integration with Oshieru

This GVHMR server integrates with the main Oshieru iOS app via:

1. **Video upload:** App uploads MP4 to storage
2. **Job queue:** Firebase/Redis queues the job
3. **Processing:** GVHMR server processes video
4. **Results:** USDZ + metrics uploaded to storage
5. **App download:** iOS app downloads USDZ for visualization

The USDZ file contains:
- Semi-transparent body mesh (α=0.25)
- Animated armature (UsdSkel)
- Bone proxy geometry (cyan spheres + sticks)
- Golf club proxy cylinder

This is displayed in the iOS app using `USDZViewerRK.swift` (RealityKit).
