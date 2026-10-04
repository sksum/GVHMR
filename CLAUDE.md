# CLAUDE.md

This file provides guidance for working with the GVHMR (Global Video-based Human Mesh Recovery) codebase in the context of the Oshieru golf swing training application.

## Project Overview

GVHMR is the video-to-3D processing pipeline that converts uploaded golf swing videos into USDZ 3D models with metrics and AI-generated coaching audio. It runs as a GPU worker service, typically deployed on Kaggle with T4 x2 GPU configuration.

**Primary entry point:** `mainserver (3).ipynb` - Jupyter notebook serving as the main processing server

**Core function:**
1. Download golf swing video from Upstash Redis queue
2. Run GVHMR to extract SMPL pose parameters (frame-by-frame 3D human pose)
3. Process results in Blender: generate USDZ with transparent body, bone proxies, and golf club proxy
4. Calculate golf swing metrics (X-factor, tempo, weight distribution, etc.)
5. Extract camera data for iOS viewer positioning
6. Generate AI coaching narration (Gemini) + audio (Fish Audio TTS)
7. Upload USDZ, metrics JSON, and MP3 back to storage

## Architecture

```
Upstash Redis (REST API over HTTP)
    ↓
worker_loop() fetches job via GET /lpop/queue:pending
    ↓
run_gvhmr() - extracts SMPL parameters (hmr4d_results.pt.pkl)
    ├── SimpleVO (default): SIFT-based visual odometry
    └── DPVO (optional): Use --use_dpvo flag
    ↓
processs_video_output() - Blender processing
    ├── Load SMPL FBX model
    ├── Apply per-frame pose/shape
    ├── Setup transparent body material (α=0.75)
    ├── Remove hands, create golf club cylinder
    ├── Create bone proxies (joint spheres + bone sticks)
    └── Export to USDZ (UsdSkel animation)
    ↓
calculate_golf_swing_metrics_from_armature()
    ├── Sample all joint positions per frame
    ├── Calculate COM, weight distribution
    ├── Compute pelvis/chest rotation, sway, lift
    ├── Detect events (setup, top, impact)
    └── Return metrics JSON with camera data
    ↓
generate_transcript() - Gemini 2.5 Flash Lite coaching narration
    ↓
generate_audio() - Fish Audio TTS MP3 synthesis
    ↓
Upload USDZ + metrics JSON + MP3 to storage
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
!source activate gvhmr

# Clone and install GVHMR
!git clone 'https://<TOKEN>@github.com/sksum/GVHMR.git'
%cd GVHMR
!source activate gvhmr && grep -v "^chumpy" requirements.txt | pip install -r /dev/stdin
!source activate gvhmr && pip install -e .

# Install Blender Python API
%pip install usd-core bpy==4.2.0 --extra-index-url https://download.blender.org/pypi/

# Install Gemini for transcript generation
!source activate gvhmr && pip install -q google-genai
```

### Model Checkpoints (HuggingFace)

Required models downloaded via aria2c:
- `SMPL_NEUTRAL.pkl` - Body model
- `SMPLX_NEUTRAL.npz` - Extended body model
- `dpvo.pth` - Dense Pose Video (optional, not used by default)
- `gvhmr_siga24_release.ckpt` - GVHMR weights
- `epoch=10-step=25000.ckpt` - HMR2
- `vitpose-h-multi-coco.pth` - Pose estimation
- `yolov8x.pt` - Object detection

```bash
!aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
  https://huggingface.co/camenduru/SMPLer-X/resolve/main/SMPL_NEUTRAL.pkl \
  -d inputs/checkpoints/body_models/smpl -o SMPL_NEUTRAL.pkl
# ... (similar for other models)
```

## Key Components

### 1. GVHMR Processing (`tools/demo/demo.py`)

Run GVHMR on video to extract pose parameters:

```bash
python tools/demo/demo.py --video="<video_path>" -s
```

**Arguments:**
- `--video`: Path to input video
- `-s / --static_cam`: Skip visual odometry for static camera
- `--use_dpvo`: Use DPVO instead of SimpleVO (default: SimpleVO)
- `--f_mm`: Focal length in mm for full-frame cameras (iPhone 15 Pro: 13/24/48/77mm)
- `--verbose`: Draw intermediate results

**Output:** `outputs/demo/<job_id>/hmr4d_results.pt.pkl`

The pickle file contains:
```python
{
    'smpl_params_global': {
        'transl': (L, 3),      # Global translation per frame
        'global_orient': (L, 3), # Root orientation (axis-angle)
        'body_pose': (L, 69),   # Joint rotations (22 joints × 3)
        'betas': (10,),         # Body shape parameters
    },
    'smpl_params_incam': { ... },     # Same, in camera coordinates
    'camera_params': {
        'R_w2c': (L, 3, 3),    # World-to-camera rotation
        't_w2c': (L, 3),       # Camera translation
        'K_fullimg': (3, 3),   # Intrinsic matrix
    }
}
```

**Camera Computation (pp_static_joint_cam approach):**

For static camera mode, camera position is computed from the relationship between incam and global coordinates:

```python
R_c2w = R_gv @ R_c.mT
t_c2w = P_w_pelvis - R_c2w @ P_c_pelvis
```

This produces a USDZViewerRK-compatible camera transform for the iOS RealityKit viewer.

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

**`setup_body_transparent_material(mesh_obj, alpha=0.75)`** - Create semi-transparent body material using Principled BSDF

**`create_bone_proxies(armature_obj, num_frames)`** - Create visualization geometry:
- UV spheres at joint heads (PROXY_RADIUS = 0.020m)
- Cylinders connecting parent→child bones (STICK_RADIUS = 0.012m)
- Cyan color (0.2, 0.6, 1.0)
- All keyframed per frame to follow armature

**BONE_CONNECTIONS** - Explicit joint pairs for stick cylinders:
```python
BONE_CONNECTIONS = [
    ('m_avg_Pelvis', 'm_avg_Spine1'),
    ('m_avg_Spine1', 'm_avg_Spine2'),
    # ... (spine, legs, arms)
]
```

**`remove_hands_and_create_cylinder(armature_obj, mesh_obj)`** - Remove hand vertices, create golf club proxy cylinder

**`apply_shoulder_ik(armature_obj, target_pos, side, frame)`** - Rotate shoulder to bring hands together (namaste pose), keeping elbow locked

**`animate_cylinder(armature_obj, cylinder, num_frames)`** - Orient cylinder 90° from arm in palm direction (like golf club), length = 0.25m

**`export_with_cylinder(...)`** - Export to USD with:
- Body mesh (transparent, α=0.75)
- Armature (as UsdSkeleton + SkelAnimation)
- Bone proxies (opaque cyan)
- Golf club cylinder

#### MTL Material Parsing

**`parse_mtl_file(mtl_path)`** - Parse MTL file for material properties

**`create_blender_material_from_mtl(material_name, mtl_data, mtl_dir_path)`** - Create node-based Blender material from MTL data

**`apply_mtl_materials(mtl_path)`** - Apply MTL materials to all mesh objects in the scene

### 3. Golf Swing Metrics

**`calculate_golf_swing_metrics_from_armature(armature_obj, fps=60, mesh_obj=None, pickle_data=None)`** - Main metrics function

#### Joint Sampling

```python
JOINTS = {
    "pelvis": "m_avg_Pelvis", "root": "m_avg_root",
    "L_hip": "m_avg_L_Hip", "R_hip": "m_avg_R_Hip",
    "L_knee": "m_avg_L_Knee", "R_knee": "m_avg_R_Knee",
    "L_ankle": "m_avg_L_Ankle", "R_ankle": "m_avg_R_Ankle",
    "L_foot": "m_avg_L_Foot", "R_foot": "m_avg_R_Foot",
    "spine1": "m_avg_Spine1", "spine2": "m_avg_Spine2", "spine3": "m_avg_Spine3",
    "neck": "m_avg_Neck", "head": "m_avg_Head",
    "L_collar": "m_avg_L_Collar", "R_collar": "m_avg_R_Collar",
    "L_shoulder": "m_avg_L_Shoulder", "R_shoulder": "m_avg_R_Shoulder",
    "L_elbow": "m_avg_L_Elbow", "R_elbow": "m_avg_R_Elbow",
    "L_wrist": "m_avg_L_Wrist", "R_wrist": "m_avg_R_Wrist",
    "L_hand": "m_avg_L_Hand", "R_hand": "m_avg_R_Hand",
}
```

#### Metrics Helper Functions

**Sampling:**
- `sample_frame_joints(armature_obj, frame_number)` - Sample all joint positions at a frame

**Center of Mass:**
- `compute_com_from_joints(joints)` - Compute COM from joint positions using segment weights
- `SEGMENT_WEIGHTS` - Approximate body segment weights (head: 0.07, torso: 0.50, etc.)

**Weight Distribution:**
- `calc_weight_distribution_from_com(joints, com)` - Calculate lead/trail foot weight % based on COM position

**Pelvis Metrics:**
- `pelvis_metrics_from_joints(joints, ref_pelvis_z)` - Rotation (°), sway (m), lift (m)

**Chest/Shoulder Metrics:**
- `chest_metrics_from_joints(joints, ref_chest_pos)` - Rotation (°), sway (m), lift (m)
- `shoulder_metrics_from_joints(joints)` - Alias for chest metrics

**X-Factor:**
- `calculate_x_factor(chest_rotation, pelvis_rotation)` - Chest rotation - Pelvis rotation

**Event Detection:**
- `detect_setup_frame(frames_joints, threshold)` - First significant hand movement
- `detect_top_frame(frames_joints)` - Minimum X-factor (maximum coil)
- `detect_impact_frame(frames_joints, setup_frame, top_frame)` - Hands closest to setup after top

**Duration & Tempo:**
- `calc_backswing_and_downswing(frames_joints)` - Return frame counts for each phase
- `calc_tempo(backswing_frames, downswing_frames)` - Ratio: backswing / downswing

#### Output Structure

```json
{
  "events": {
    "setup_frame": N,
    "top_frame": M,
    "impact_frame": K,
    "backswing_duration_frames": ...,
    "downswing_duration_frames": ...,
    "backswing_duration_seconds": ...,
    "downswing_duration_seconds": ...
  },
  "tempo_ratio": float,
  "per_frame_metrics": {
    "0": {
      "weight_distribution": {"lead_foot_percent": ..., "trail_foot_percent": ...},
      "pelvis": {"rotation_degrees": ..., "sway_meters": ..., "lift_meters": ...},
      "chest": {"rotation_degrees": ..., "sway_meters": ..., "lift_meters": ...},
      "x_factor_degrees": ...,
      "com_projection": {"x": ..., "y": ...},
      "ground_plane": {"pelvis": {"x": ..., "y": ..., "z": ...}, ...}
    },
    ...
  },
  "summary_metrics": {
    "max_x_factor": ...,
    "min_x_factor": ...,
    "x_factor_range": ...,
    "max_pelvis_rotation": ...,
    "max_chest_rotation": ...,
    "peak_com_sway": ...
  },
  "metadata": {
    "total_frames": ...,
    "fps": 60,
    "processing_timestamp": "ISO-8601",
    "bone_mapping": "smplx_standard"
  },
  "camera": {
    "position": [x, y, z],
    "rotation": [w, x, y, z],
    "mode": "static_incam_derived",
    "method": "pp_static_joint_cam"
  }
}
```

### 4. Camera Data Extraction

**`extract_camera_data_from_pickle(pickle_data)`** - Extract camera transform for iOS viewer

Uses the `pp_static_joint_cam` approach from `hmr4d/model/gvhmr/utils/postprocess.py`:

```python
# From incam/global relationship
R_c2w = R_gv @ R_c.T
t_c2w = P_w_pelvis - R_c2w @ P_c_pelvis

# Convert to USDZViewerRK format
{
  'position': [x, y, z],
  'rotation': [w, x, y, z],  # Quaternion
  'mode': 'static_incam_derived'
}
```

### 5. Gemini Transcript + Fish Audio TTS (Cell 8)

**`generate_transcript(metrics_json, preferences)`** - Generate multi-chapter coaching narration

- Uses Gemini 2.5 Flash Lite via `google-genai` SDK
- Prompt template includes available visualization chapters (full_swing, pelvis_arc, etc.)
- Outputs plain English with `[ANIM:START id="<CHAPTER_ID>"]` markers

**`generate_audio(transcript)`** - Convert transcript to MP3

- Uses Fish Audio TTS REST API
- Returns raw MP3 bytes
- Reference ID: `"bf322df2096a46f18c579d0baa36f41d"`

### 6. Worker Loop

Multi-GPU processing with Upstash Redis queue:

```python
NUM_GPUS = 2  # Kaggle T4 x2

def redis_get(path):
    requests.get(f"{UPSTASH_URL}/{path}",
                headers={"Authorization": f"Bearer {UPSTASH_TOKEN}"})

def redis_set(key, value):
    requests.post(f"{UPSTASH_URL}/set/{key}",
                 headers={"Authorization": f"Bearer {UPSTASH_TOKEN}"},
                 data=json.dumps(value))

def worker_loop(gpu_id, job_queue):
    while True:
        job = job_queue.get()
        # Download video
        # Run GVHMR with CUDA_VISIBLE_DEVICES=gpu_id
        # Process in Blender
        # Calculate metrics
        # Generate transcript + audio
        # Upload USDZ + JSON + MP3 via PUT to presigned URLs
        # Update Redis status
```

**Upstash REST API:**
- GET/SET operations via HTTP (not Redis client library)
- Bearer token authentication
- Job queue: `GET /lpop/queue:pending`
- Job data: `GET /get/job:{job_id}`, `POST /set/job:{job_id}`

**Upload Contract:**
```python
job["worker_urls"] = {
    "download_mp4": "...",
    "upload_usdz": "...",
    "upload_json": "...",
    "upload_mp3": "..."
}
```

## File Structure

```
GVHMR/
├── mainserver (3).ipynb          # Main processing server (THIS FILE)
├── hmr4d/                        # Core GVHMR library
│   ├── configs/                  # Hydra configs
│   ├── datamodule/               # Data modules
│   ├── dataset/                  # Datasets (bedlam, emdb, h36m, etc.)
│   ├── model/                    # Model implementations
│   ├── network/                  # Network architectures
│   └── utils/                    # Utilities
│       ├── body_model/           # SMPL/SMPLX models
│       ├── geo/                  # Geometry utilities
│       ├── preproc/              # Preprocessing (SimpleVO, VitPose)
│       └── vis/                  # Visualization
├── tools/
│   ├── demo/
│   │   ├── demo.py              # GVHMR inference script
│   │   ├── demo_folder.py       # Batch inference
│   │   └── colab_demo.ipynb     # Colab demo
│   ├── train.py                  # Training/reproduction script
│   └── unitest/                 # Unit tests
├── inputs/
│   └── checkpoints/             # Model weights (created at runtime)
│       ├── body_models/
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

# Calculate metrics
armature_obj = bpy.data.objects.get('m_avg')
results = calculate_golf_swing_metrics_from_armature(armature_obj, fps=60)

# Outputs:
# - /kaggle/working/outputs/{job_id}.usdz
# - metrics JSON (with camera data)
```

## Known Issues & Notes

- **GPU requirement:** GVHMR requires CUDA GPU. T4 x2 on Kaggle is recommended configuration.
- **Memory:** Processing multiple jobs concurrently requires significant GPU memory.
- **Blender version:** Uses `bpy==4.2.0` for USD export with ARKit compatibility.
- **Coordinate system:** Y-up coordinate system; `HIGH_FROM_FLOOR = 1.5m` offset.
- **Hand removal:** Hands are removed from mesh and replaced with cylinder proxy for golf club visualization.
- **Shoulder IK:** Uses simplified IK that only rotates shoulder (elbow locked) to bring hands to midpoint.
- **SimpleVO vs DPVO:** SimpleVO (SIFT-based) is default since March 2025. DPVO is opt-in via `--use_dpvo`.
- **DPVO submodule:** `third-party/DPVO/` is declared in `.gitmodules` but not checked out (empty).
- **Scene FPS:** Must set `scene.render.fps = 60` before animation to ensure USDZ timing matches iOS expectations.

## Dependencies

Key Python packages (from `requirements.txt`):

```txt
# PyTorch (CUDA 12.1)
torch==2.3.0+cu121
torchvision==0.18.0+cu121

# Lightning + Hydra
lightning==2.3.0
hydra-core==1.3
hydra-zen
hydra_colorlog
rich

# Common utilities
numpy==1.23.5
jupyter
matplotlib
opencv-python
ffmpeg-python
scikit-image
einops
imageio==2.34.1
av==13.0.0
joblib

# 3D Vision
pytorch3d @ https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt230/pytorch3d-0.7.6-cp310-cp310-linux_x86_64.whl
trimesh
chumpy
smplx
wis3d
pycolmap

# 2D Pose
ultralytics==8.2.42  # YOLO
cython_bbox
lapx

# HMR2.0a feature extraction
timm==0.9.12
```

Additional packages installed separately:
- `bpy==4.2.0` - Blender Python API
- `usd-core` - Pixar USD library
- `google-genai` - Gemini API for transcript generation

## Integration with Oshieru

This GVHMR server integrates with the main Oshieru iOS app via:

1. **Video upload:** App uploads MP4 to storage
2. **Job queue:** Upstash Redis queues the job (REST API)
3. **Processing:** GVHMR server processes video
4. **Results:** USDZ + metrics JSON + MP3 uploaded to storage
5. **App download:** iOS app downloads USDZ for visualization

The USDZ file contains:
- Semi-transparent body mesh (α=0.75)
- Animated armature (UsdSkel)
- Bone proxy geometry (cyan spheres + sticks)
- Golf club proxy cylinder

The metrics JSON contains:
- Per-frame metrics (weight distribution, pelvis/chest rotation, X-factor)
- Event frames (setup, top, impact)
- Tempo ratio
- Camera data for iOS viewer positioning
- Coaching transcript (for fallback)

The MP3 file contains:
- AI-generated golf swing narration (Gemini 2.5 Flash Lite)
- Text-to-speech synthesis (Fish Audio TTS)

This is displayed in the iOS app using `USDZViewerRK.swift` (RealityKit) with chapter-based animation markers.

## Output Artifacts

| Artifact | Format | Description |
|----------|--------|-------------|
| USDZ | `model/vnd.usdz+zip` | 3D model with animation |
| Metrics JSON | `application/json` | Swing analysis + camera data |
| MP3 | `audio/mpeg` | AI coaching narration |
