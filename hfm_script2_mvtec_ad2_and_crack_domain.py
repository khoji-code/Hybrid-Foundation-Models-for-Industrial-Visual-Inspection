#  "Industrial Visual Inspection Using Self-Supervised, Few-Shot and Zero-Shot Learning"
#  Authors: Hedieh Sajedi, Abolfazl Khojasteh Abkenar

import subprocess, sys, os

def _pip_install(packages):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + packages)
    except Exception as e:
        print("[warn] pip install problem:", e)

if not os.environ.get("HFM_MOCK"):
    _pip_install(["transformers", "scikit-learn", "scipy", "pandas", "matplotlib", "tabulate",
                  "kaggle", "psutil", "pillow", "tqdm"])

import gc, json, math, time, pickle, shutil, random, itertools, warnings, platform, io, glob, zipfile, traceback
from collections import defaultdict, OrderedDict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as tvm
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFilter, ImageEnhance
from scipy import ndimage as ndi
from scipy import stats as sps
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
warnings.filterwarnings("ignore")



SCRIPT_ID = 2
DATASETS_TO_RUN = ["mvtec_ad_2", "crack"]
ROBUSTNESS_DATASETS = []          
BACKBONE_SCALING_DATASETS = []   
RUN_EFFICIENCY = False           

QUICK_TEST = bool(int(os.environ.get("HFM_QUICK", "0")))   
ROOT = os.environ.get("HFM_ROOT", "/content")
OUT_DIR = f"{ROOT}/hfm_outputs_script{SCRIPT_ID}"
CACHE_DIR = f"{ROOT}/hfm_cache"
ZIP_PATH = f"{ROOT}/output_script{SCRIPT_ID}.zip"
USE_DRIVE_BACKUP = False        
BASE_DATASETS_DIR = f"{ROOT}/industrial_datasets"

N_SEEDS = 10                    
SEED_BASE = 1000
TRAIN_CAP = 500                 
VAL_FRAC = 0.20                 
MIN_VAL = 8
CAL_DEF_FRAC = 0.20             
MIN_CAL_DEF = 2
THR_QUANTILES = (95, 99)        
LOC_SEEDS = 3                   
ROB_SEEDS = 3
BANK_SEEDS = 3
FEWSHOT_KS = [1, 5, 10]         
NORMAL_SHOT_KS = [1, 2, 4, 10]  
BANK_RATIOS = [0.02, 0.05, 0.15, 0.30]
ALPHAS = [1, 10, 50]            
PIX_Q = 99.5                    
SMOOTH_SIGMA = 4.0
ROB_CORRUPTIONS = ["brightness_low", "brightness_high", "gaussian_noise", "gaussian_blur", "jpeg_q20", "rotation_8deg"]
BOOT_N = 10000

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
IMG_SIZE = 336
RESIZE_SIZE = 384
PATCH = 14
GRID = IMG_SIZE // PATCH        # 24
NP_DINO = GRID * GRID           # 576
NUM_DINO_LAYERS = 4
GOOD_NAMES = ("good", "ok", "normal")
MASK_DIR_NAMES = ("ground_truth", "gt", "masks")
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IM_MEAN = (0.485, 0.456, 0.406)
IM_STD = (0.229, 0.224, 0.225)
BANK_RATIO = 0.15
BANK_POOL_MAX = 40000
TOPK_IMG = 3
NUM_WORKERS = 2
BATCH = 16

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(os.path.join(CACHE_DIR, "results"), exist_ok=True)
for _d in ("tables", "figures", "raw"):
    os.makedirs(os.path.join(OUT_DIR, _d), exist_ok=True)

_LOG_PATH = os.path.join(OUT_DIR, "run_log.txt")
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    try:
        with open(_LOG_PATH, "a") as f:
            f.write(s + "\n")
    except Exception:
        pass

def set_global_seed(s=0):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def env_report():
    info = OrderedDict()
    info["script_id"] = SCRIPT_ID
    info["python"] = platform.python_version()
    info["torch"] = torch.__version__
    info["torchvision"] = torchvision.__version__
    info["numpy"] = np.__version__
    info["pandas"] = pd.__version__
    try:
        import transformers, sklearn, scipy
        info["transformers"] = transformers.__version__
        info["sklearn"] = sklearn.__version__
        info["scipy"] = scipy.__version__
    except Exception:
        pass
    info["device"] = str(DEVICE)
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_mem_GB"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    try:
        import psutil
        info["ram_GB"] = round(psutil.virtual_memory().total / 1e9, 1)
        info["cpu_count"] = os.cpu_count()
    except Exception:
        pass
    info["config"] = {k: globals()[k] for k in
        ["N_SEEDS", "SEED_BASE", "TRAIN_CAP", "VAL_FRAC", "CAL_DEF_FRAC", "THR_QUANTILES", "LOC_SEEDS", "ROB_SEEDS",
         "BANK_SEEDS", "FEWSHOT_KS", "NORMAL_SHOT_KS", "BANK_RATIOS", "ALPHAS", "PIX_Q", "SMOOTH_SIGMA",
         "IMG_SIZE", "RESIZE_SIZE", "BANK_RATIO", "BANK_POOL_MAX", "TOPK_IMG", "DATASETS_TO_RUN", "QUICK_TEST"]}
    return info

PAL = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "verm": "#D55E00",
       "sky": "#56B4E9", "pink": "#CC79A7", "yellow": "#F0E442", "grey": "#666666", "black": "#000000"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 11, "axes.labelsize": 10, "legend.frameon": False,
                     "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight"})

def save_fig(fig, name, dpi=250):
    p = os.path.join(OUT_DIR, "figures", name)
    fig.savefig(p, dpi=dpi)
    plt.close(fig)
    log("  saved figure:", name)
    return p

def save_table(df, name, index=False):
    p = os.path.join(OUT_DIR, "tables", name)
    df.to_csv(p, index=index)
    log("  saved table:", name, f"({len(df)} rows)")
    return p

MVTEC_KAGGLE_SLUG = "ipythonx/mvtec-ad"
BTAD_KAGGLE_SLUG = "thtuan/btad-beantech-anomaly-detection"
MPDD_KAGGLE_SLUG = "lephonghao/metal-parts-defect-detection-dataset-mpdd"
CRACK_KAGGLE_SLUGS = ["arunrk7/surface-crack-detection", "thesenimark/concrete-crack-images-for-classification"]
VISA_TAR_URL = "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"
VISA_SPLIT_CSV_URL = "https://raw.githubusercontent.com/amazon-science/spot-diff/main/split_csv/1cls.csv"
MVTEC_AD2_TAR_URL = "https://www.mydrive.ch/shares/150997/701c90d3aea6588f404936e32a674602/download/466712769-1743429042/mvtec_ad_2.tar.gz"
MVTEC_AD2_MERGE_VALIDATION_INTO_TRAIN = True
CRACK_N_TRAIN, CRACK_N_TEST_EACH = 1000, 500

def ensure_kaggle_credentials():
    if os.path.exists(os.path.expanduser('~/.kaggle/kaggle.json')):
        return
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return
    try:
        from google.colab import files
        print("Upload your kaggle.json (Kaggle account -> Settings -> Create New API Token):")
        files.upload()
        os.system("mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json")
    except Exception as e:
        raise RuntimeError("Kaggle credentials not found (put kaggle.json in ~/.kaggle).") from e

def find_category_dirs_generic(root):
    cats = []
    for dirpath, dirnames, _ in os.walk(root):
        if "train" in dirnames and "test" in dirnames:
            train_dir = os.path.join(dirpath, "train")
            sub = [d.lower() for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
            if any(s in GOOD_NAMES for s in sub):
                cats.append(dirpath)
    return sorted(cats)

def find_mask_dir(cat_dir):
    for name in MASK_DIR_NAMES:
        p = os.path.join(cat_dir, name)
        if os.path.isdir(p):
            return p
    return None

def get_mvtec_style_paths_with_masks(cat_dir):
    train_root = os.path.join(cat_dir, "train")
    good_name = next((c for c in os.listdir(train_root) if c.lower() in GOOD_NAMES
                      and os.path.isdir(os.path.join(train_root, c))), None)
    train_paths = []
    if good_name:
        gd = os.path.join(train_root, good_name)
        train_paths = [os.path.join(gd, f) for f in sorted(os.listdir(gd)) if f.lower().endswith(IMG_EXTS)]
    mask_root = find_mask_dir(cat_dir)
    test_root = os.path.join(cat_dir, "test")
    test_paths, test_labels, mask_paths, defect_types = [], [], [], []
    if os.path.isdir(test_root):
        for defect_type in sorted(os.listdir(test_root)):
            d = os.path.join(test_root, defect_type)
            if not os.path.isdir(d):
                continue
            is_good = defect_type.lower() in GOOD_NAMES
            label = 0 if is_good else 1
            for f in sorted(os.listdir(d)):
                if not f.lower().endswith(IMG_EXTS):
                    continue
                test_paths.append(os.path.join(d, f))
                test_labels.append(label)
                defect_types.append("good" if is_good else defect_type)
                mp = None
                if (not is_good) and mask_root is not None:
                    stem = os.path.splitext(f)[0]
                    cand = [os.path.join(mask_root, defect_type, stem + "_mask" + e) for e in IMG_EXTS] + \
                           [os.path.join(mask_root, defect_type, stem + e) for e in IMG_EXTS]
                    for c in cand:
                        if os.path.exists(c):
                            mp = c
                            break
                mask_paths.append(mp)
    return train_paths, test_paths, test_labels, mask_paths, defect_types

def _download_kaggle_dataset(folder_name, slug):
    root = os.path.join(BASE_DATASETS_DIR, folder_name)
    os.makedirs(BASE_DATASETS_DIR, exist_ok=True)
    if find_category_dirs_generic(root):
        log(f"{folder_name} already present.")
        return root
    log(f"{folder_name} not found. Downloading from Kaggle ({slug})...")
    ensure_kaggle_credentials()
    os.system(f"kaggle datasets download -d {slug} -p {root} --unzip")
    if not find_category_dirs_generic(root):
        raise RuntimeError(f"{folder_name} download failed or unexpected layout (expected train/{{good,ok,normal}}, test/...).")
    return root

def collect_kaggle_style_tasks(dataset_name, root):
    tasks = []
    for cat_dir in find_category_dirs_generic(root):
        name = os.path.basename(cat_dir)
        train_p, test_p, test_y, mask_p, dtypes = get_mvtec_style_paths_with_masks(cat_dir)
        if train_p and test_p and len(set(test_y)) == 2:
            tasks.append((dataset_name, name, train_p, test_p, test_y, mask_p, dtypes))
    return tasks

def download_visa():
    root = os.path.join(BASE_DATASETS_DIR, "VisA")
    os.makedirs(root, exist_ok=True)
    extracted = any(os.path.isdir(os.path.join(root, d, "Data", "Images")) for d in os.listdir(root)
                    if os.path.isdir(os.path.join(root, d)))
    if not extracted:
        log("VisA not found. Downloading tar (~2 GB)...")
        tar_path = os.path.join(BASE_DATASETS_DIR, "VisA_20220922.tar")
        os.system(f"wget -q {VISA_TAR_URL} -O {tar_path}")
        if not os.path.exists(tar_path) or os.path.getsize(tar_path) == 0:
            raise RuntimeError("VisA tar download failed.")
        os.system(f"tar -xf {tar_path} -C {root}")
        try:
            os.remove(tar_path)
        except Exception:
            pass
    csv_path = os.path.join(root, "split_csv_1cls.csv")
    if not os.path.exists(csv_path):
        os.system(f"wget -q {VISA_SPLIT_CSV_URL} -O {csv_path}")
    return root

def load_visa_split_csv(visa_root):
    csv_path = os.path.join(visa_root, "split_csv_1cls.csv")
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        raise RuntimeError("Official VisA split CSV not available; the revision requires the official split.")
    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]
    if not {"object", "split", "label", "image"}.issubset(df.columns):
        raise RuntimeError("Unexpected VisA split CSV format.")
    return df

def get_visa_paths(visa_root, category_name, split_df):
    sub = split_df[split_df["object"] == category_name]
    train_paths, test_paths, test_labels, mask_paths, defect_types = [], [], [], [], []
    for _, r in sub.iterrows():
        full = os.path.join(visa_root, r["image"])
        if str(r["split"]).lower() == "train":
            train_paths.append(full)
        else:
            is_anom = str(r["label"]).lower() != "normal"
            test_paths.append(full); test_labels.append(1 if is_anom else 0)
            defect_types.append("anomaly" if is_anom else "good")
            mp = None
            if is_anom:
                if "mask" in sub.columns and isinstance(r["mask"], str) and len(r["mask"]) > 0:
                    cand = os.path.join(visa_root, r["mask"])
                    if os.path.exists(cand):
                        mp = cand
                if mp is None:
                    stem = os.path.splitext(os.path.basename(r["image"]))[0]
                    for e in IMG_EXTS:
                        cand = os.path.join(visa_root, category_name, "Data", "Masks", "Anomaly", stem + e)
                        if os.path.exists(cand):
                            mp = cand; break
            mask_paths.append(mp)
    return train_paths, test_paths, test_labels, mask_paths, defect_types

def collect_visa_tasks():
    root = download_visa()
    split_df = load_visa_split_csv(root)
    cats = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d, "Data", "Images")))
    tasks = []
    for name in cats:
        tr, te, y, mk, dt = get_visa_paths(root, name, split_df)
        if tr and te and len(set(y)) == 2:
            n_m = sum(1 for m, yy in zip(mk, y) if yy == 1 and m is not None)
            log(f"  [visa] {name}: {len(tr)} train, {len(te)} test, masks matched for {n_m}/{sum(y)} defective")
            tasks.append(("visa", name, tr, te, y, mk, dt))
    return tasks

def _list_imgs(d):
    if not os.path.isdir(d):
        return []
    return sorted(f for f in os.listdir(d) if f.lower().endswith(IMG_EXTS))

def discover_mvtec_ad2_categories(root):
    if not os.path.isdir(root):
        return []
    return sorted(n for n in os.listdir(root) if os.path.isdir(os.path.join(root, n, "train", "good")))

def find_mvtec_ad2_mask(gt_dir, image_filename):
    if not os.path.isdir(gt_dir):
        return None
    stem = os.path.splitext(image_filename)[0]
    search_dirs = [gt_dir, os.path.join(gt_dir, "bad")]
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for e in IMG_EXTS:
            for cand in (os.path.join(d, stem + "_mask" + e), os.path.join(d, stem + e)):
                if os.path.exists(cand):
                    return cand
    for d in search_dirs:
        if os.path.isdir(d):
            m = sorted(x for x in glob.glob(os.path.join(d, stem + "*")) if x.lower().endswith(IMG_EXTS))
            if m:
                return m[0]
    return None

def get_mvtec_ad2_category_paths(cat_dir, merge_validation_into_train=True):
    train_dir = os.path.join(cat_dir, "train", "good")
    val_dir = os.path.join(cat_dir, "validation", "good")
    train_paths = [os.path.join(train_dir, f) for f in _list_imgs(train_dir)]
    if merge_validation_into_train:
        train_paths += [os.path.join(val_dir, f) for f in _list_imgs(val_dir)]
    tp = os.path.join(cat_dir, "test_public")
    good_dir, bad_dir, gt_dir = [os.path.join(tp, x) for x in ("good", "bad", "ground_truth")]
    test_paths, test_labels, mask_paths, defect_types = [], [], [], []
    for f in _list_imgs(good_dir):
        test_paths.append(os.path.join(good_dir, f)); test_labels.append(0); mask_paths.append(None); defect_types.append("good")
    for f in _list_imgs(bad_dir):
        test_paths.append(os.path.join(bad_dir, f)); test_labels.append(1); defect_types.append("bad")
        mask_paths.append(find_mvtec_ad2_mask(gt_dir, f))
    return train_paths, test_paths, test_labels, mask_paths, defect_types

def download_mvtec_ad2():
    parent = os.path.join(BASE_DATASETS_DIR, "mvtec_ad_2_download")
    root = os.path.join(parent, "mvtec_ad_2")
    os.makedirs(parent, exist_ok=True)
    if discover_mvtec_ad2_categories(root):
        log("MVTec AD 2 already present."); return root
    if discover_mvtec_ad2_categories(parent):
        return parent
    log("MVTec AD 2 not found. Downloading tar.gz (large, may take a long time)...")
    tar_path = os.path.join(parent, "mvtec_ad_2.tar.gz")
    os.system(f'wget -q --no-check-certificate -U "Mozilla/5.0" "{MVTEC_AD2_TAR_URL}" -O {tar_path}')
    if not os.path.exists(tar_path) or os.path.getsize(tar_path) == 0:
        raise RuntimeError("MVTec AD 2 download failed.")
    log("Extracting...")
    os.system(f"tar -xzf {tar_path} -C {parent}")
    try:
        os.remove(tar_path)      
    except Exception:
        pass
    if discover_mvtec_ad2_categories(root):
        return root
    if discover_mvtec_ad2_categories(parent):
        return parent
    raise RuntimeError("MVTec AD 2 extraction failed or unexpected layout.")

def collect_mvtec_ad2_tasks():
    root = download_mvtec_ad2()
    tasks = []
    for cat in discover_mvtec_ad2_categories(root):
        tr, te, y, mk, dt = get_mvtec_ad2_category_paths(os.path.join(root, cat), MVTEC_AD2_MERGE_VALIDATION_INTO_TRAIN)
        if tr and te and len(set(y)) == 2:
            n_m = sum(1 for m, yy in zip(mk, y) if yy == 1 and m is not None)
            log(f"  [mvtec_ad_2] {cat}: {len(tr)} train, {len(te)} test_public ({y.count(0)} good / {y.count(1)} bad), masks {n_m}")
            tasks.append(("mvtec_ad_2", cat, tr, te, y, mk, dt))
    return tasks

def collect_crack_tasks():
    root = os.path.join(BASE_DATASETS_DIR, "crack")
    def _find(root):
        pos = neg = None
        for dp, dn, _ in os.walk(root):
            for d in dn:
                if d.lower() == "positive": pos = os.path.join(dp, d)
                if d.lower() == "negative": neg = os.path.join(dp, d)
        return pos, neg
    pos, neg = _find(root) if os.path.isdir(root) else (None, None)
    if pos is None or neg is None:
        ensure_kaggle_credentials()
        os.makedirs(root, exist_ok=True)
        for slug in CRACK_KAGGLE_SLUGS:
            os.system(f"kaggle datasets download -d {slug} -p {root} --unzip")
            pos, neg = _find(root)
            if pos and neg:
                break
    if pos is None or neg is None:
        raise RuntimeError("Concrete-crack dataset could not be downloaded (extra-domain study skipped).")
    rng = np.random.RandomState(0)
    neg_f = sorted(os.path.join(neg, f) for f in os.listdir(neg) if f.lower().endswith(IMG_EXTS))
    pos_f = sorted(os.path.join(pos, f) for f in os.listdir(pos) if f.lower().endswith(IMG_EXTS))
    neg_f = [neg_f[i] for i in rng.permutation(len(neg_f))]
    pos_f = [pos_f[i] for i in rng.permutation(len(pos_f))]
    train = neg_f[:CRACK_N_TRAIN]
    test_neg = neg_f[CRACK_N_TRAIN:CRACK_N_TRAIN + CRACK_N_TEST_EACH]
    test_pos = pos_f[:CRACK_N_TEST_EACH]
    test_p = test_neg + test_pos
    y = [0] * len(test_neg) + [1] * len(test_pos)
    dt = ["good"] * len(test_neg) + ["crack"] * len(test_pos)
    log(f"  [crack] {len(train)} normal train, {len(test_neg)} normal test, {len(test_pos)} cracked test")
    return [("crack", "concrete_surface", train, test_p, y, [None] * len(test_p), dt)]

def collect_all_dataset_tasks():
    all_tasks = []
    for ds in DATASETS_TO_RUN:
        try:
            log(f"\nPreparing dataset: {ds}")
            if ds == "mvtec":
                t = collect_kaggle_style_tasks("mvtec", _download_kaggle_dataset("MVTec", MVTEC_KAGGLE_SLUG))
            elif ds == "btad":
                t = collect_kaggle_style_tasks("btad", _download_kaggle_dataset("BTAD", BTAD_KAGGLE_SLUG))
            elif ds == "mpdd":
                t = collect_kaggle_style_tasks("mpdd", _download_kaggle_dataset("MPDD", MPDD_KAGGLE_SLUG))
            elif ds == "visa":
                t = collect_visa_tasks()
            elif ds == "mvtec_ad_2":
                t = collect_mvtec_ad2_tasks()
            elif ds == "crack":
                t = collect_crack_tasks()
            else:
                t = []
            log(f"  -> {len(t)} categories for {ds}")
            if QUICK_TEST:
                t = t[:2]
            all_tasks.extend(t)
        except Exception as e:
            log(f"[ERROR] dataset {ds} could not be prepared: {e}")
            log(traceback.format_exc())
    log(f"\nTotal categories: {len(all_tasks)}")
    return all_tasks

def object_name(ds, cat):
    c = cat.replace("_", " ")
    if ds == "btad":
        return "industrial product"
    if ds == "crack":
        return "concrete surface"
    if ds == "visa":
        return c.rstrip("0123456789").strip() or c
    return c

if os.environ.get("HFM_MOCK"):
    NUM_WORKERS = 0

class DinoNet:
    def __init__(self, hub_name="dinov2_vitl14", fp16=True):
        self.name, self.fp16 = hub_name, fp16
        m = torch.hub.load('facebookresearch/dinov2', hub_name).to(DEVICE).eval()
        if fp16 and DEVICE.type == "cuda":
            m = m.half()
        else:
            self.fp16 = False
        self.model = m
        self.dim = m.embed_dim
        for p in self.model.parameters():
            p.requires_grad_(False)
    @torch.no_grad()
    def __call__(self, x):
        if self.fp16:
            x = x.half()
        outs = self.model.get_intermediate_layers(x, n=NUM_DINO_LAYERS, reshape=False, return_class_token=True)
        grid = torch.stack([o[0] for o in outs], 0).float().mean(0)      # (B, 576, D)  eq. (2)
        cls = outs[-1][1].float()                                         # (B, D)       eq. (1)
        return cls, grid

class ClipNet:
    def __init__(self):
        from transformers import CLIPModel, CLIPProcessor
        self.model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE).eval()
        self.proc = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        self.dim = self.model.config.projection_dim
        for p in self.model.parameters():
            p.requires_grad_(False)
    @torch.no_grad()
    def image_embed(self, px):
        out = self.model.vision_model(pixel_values=px)
        f = self.model.visual_projection(out.pooler_output).float()
        return f / f.norm(dim=-1, keepdim=True)
    @torch.no_grad()
    def text_embed(self, prompts):
        inp = self.proc(text=list(prompts), return_tensors="pt", padding=True).to(DEVICE)
        out = self.model.text_model(input_ids=inp["input_ids"], attention_mask=inp.get("attention_mask"))
        f = self.model.text_projection(out.pooler_output).float()
        return (f / f.norm(dim=-1, keepdim=True)).cpu()

class ResNetGlobal:
    def __init__(self):
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Identity()
        self.model = m.to(DEVICE).eval()
        self.dim = 2048
    @torch.no_grad()
    def __call__(self, x):
        return self.model(x).float()

class WRNPatch:
    def __init__(self):
        m = tvm.wide_resnet50_2(weights=tvm.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        self.model = m.to(DEVICE).eval()
        self.dim = 1024
        self.n_patches = 28 * 28
    @staticmethod
    def _proj(f):
        b, c, h, w = f.shape
        t = f.permute(0, 2, 3, 1).reshape(-1, 1, c)
        t = F.adaptive_avg_pool1d(t, 1024)
        return t.reshape(b, h * w, 1024)
    @torch.no_grad()
    def __call__(self, x):
        m = self.model
        h = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        h = m.layer1(h); f2 = m.layer2(h); f3 = m.layer3(f2)
        f2 = F.avg_pool2d(f2, 3, 1, 1); f3 = F.avg_pool2d(f3, 3, 1, 1)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        return (self._proj(f2) + self._proj(f3)).div(2.0)          # (B, 784, 1024)

class Nets:
    pass

def load_nets():
    nets = Nets()
    log("Loading DINOv2 ViT-L/14 (frozen, FP16)...")
    nets.dino = DinoNet("dinov2_vitl14", fp16=True)
    log("Loading CLIP ViT-L/14 (frozen, FP32)...")
    nets.clip = ClipNet()
    log("Loading ResNet-50 and Wide-ResNet-50-2 (ImageNet weights)...")
    nets.r50 = ResNetGlobal()
    nets.wrn = WRNPatch()
    return nets

# ---------------- image geometry: identical for images and masks ----------------
def shorter_size(w, h, s):
    if w <= h:
        return s, int(s * h / w)
    return int(s * w / h), s

def resize_shorter(img, s, resample):
    w, h = img.size
    nw, nh = shorter_size(w, h, s)
    if (nw, nh) == (w, h):
        return img
    gap = 2.0 if min(w, h) > 3 * s else None
    return img.resize((nw, nh), resample, reducing_gap=gap)

def center_crop(img, size):
    w, h = img.size
    l = int(round((w - size) / 2.0)); t = int(round((h - size) / 2.0))
    return img.crop((l, t, l + size, t + size))

def pil_to_u8(img):
    return torch.from_numpy(np.asarray(img, dtype=np.uint8).copy()).permute(2, 0, 1)

def load_gt_mask(mask_path, mode="aligned"):
    m = Image.open(mask_path).convert("L")
    if mode == "aligned":
        m = center_crop(resize_shorter(m, RESIZE_SIZE, Image.NEAREST), IMG_SIZE)
        return (np.array(m) > 0).astype(np.uint8)
    m = m.resize((IMG_SIZE, IMG_SIZE))
    return (np.array(m) > 127).astype(np.uint8)

def apply_corruption(img, name, idx):
    img = resize_shorter(img, 512, Image.BILINEAR)
    if name == "brightness_low":
        return ImageEnhance.Brightness(img).enhance(0.5)
    if name == "brightness_high":
        return ImageEnhance.Brightness(img).enhance(1.6)
    if name == "gaussian_noise":
        rng = np.random.RandomState(7919 + idx)
        a = np.asarray(img, dtype=np.float32) + rng.normal(0, 0.05 * 255, size=(img.size[1], img.size[0], 3))
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if name == "gaussian_blur":
        return img.filter(ImageFilter.GaussianBlur(radius=2))
    if name == "jpeg_q20":
        buf = io.BytesIO(); img.save(buf, format="JPEG", quality=20); buf.seek(0)
        return Image.open(buf).convert("RGB")
    if name == "rotation_8deg":
        fill = tuple(int(v) for v in np.asarray(img).reshape(-1, 3).mean(0))
        return img.rotate(8, resample=Image.BILINEAR, fillcolor=fill)
    raise ValueError(name)

class MultiViewDataset(Dataset):
    def __init__(self, paths, corruption=None, views=("dino", "clip", "cnn")):
        self.paths, self.corruption, self.views = list(paths), corruption, views
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        try:
            img = Image.open(self.paths[i]).convert("RGB")
        except Exception as e:
            print(f"[warn] unreadable image {self.paths[i]}: {e}")
            img = Image.new("RGB", (RESIZE_SIZE, RESIZE_SIZE))
        if self.corruption:
            img = apply_corruption(img, self.corruption, i)
        d = c = r = torch.zeros(1, dtype=torch.uint8)
        if "dino" in self.views:
            d = pil_to_u8(center_crop(resize_shorter(img, RESIZE_SIZE, Image.BILINEAR), IMG_SIZE))
        if "clip" in self.views:
            c = pil_to_u8(center_crop(resize_shorter(img, 224, Image.BICUBIC), 224))
        if "cnn" in self.views:
            r = pil_to_u8(center_crop(resize_shorter(img, 256, Image.BILINEAR), 224))
        return d, c, r

def _norm(x_u8, mean, std):
    x = x_u8.to(DEVICE, non_blocking=True).float().div_(255.0)
    m = torch.tensor(mean, device=DEVICE).view(1, 3, 1, 1)
    s = torch.tensor(std, device=DEVICE).view(1, 3, 1, 1)
    return (x - m) / s

@torch.no_grad()
def extract_features(paths, nets, want=("dino", "clip", "r50", "wrn"), corruption=None, desc="features", dino_nets=None):
    N = len(paths)
    out = {}
    if "dino" in want:
        out["cls"] = torch.empty((N, nets.dino.dim), dtype=torch.float32)
        out["grid"] = torch.empty((N, NP_DINO, nets.dino.dim), dtype=torch.float16)
    if "clip" in want:
        out["clip"] = torch.empty((N, nets.clip.dim), dtype=torch.float32)
    if "r50" in want:
        out["r50"] = torch.empty((N, nets.r50.dim), dtype=torch.float32)
    if "wrn" in want:
        out["wgrid"] = torch.empty((N, nets.wrn.n_patches, nets.wrn.dim), dtype=torch.float16)
    for nm, dn in (dino_nets or {}).items():
        out[f"cls_{nm}"] = torch.empty((N, dn.dim), dtype=torch.float32)
        out[f"grid_{nm}"] = torch.empty((N, NP_DINO, dn.dim), dtype=torch.float16)
    views = []
    if "dino" in want or dino_nets: views.append("dino")
    if "clip" in want: views.append("clip")
    if "r50" in want or "wrn" in want: views.append("cnn")
    loader = DataLoader(MultiViewDataset(paths, corruption, tuple(views)), batch_size=BATCH, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"))
    i = 0
    for d, c, r in tqdm(loader, desc=desc, leave=False):
        b = d.shape[0] if "dino" in views else (c.shape[0] if "clip" in views else r.shape[0])
        if "dino" in views:
            xd = _norm(d, IM_MEAN, IM_STD)
            if "dino" in want:
                cls, grid = nets.dino(xd)
                out["cls"][i:i + b] = cls.cpu(); out["grid"][i:i + b] = grid.half().cpu()
            for nm, dn in (dino_nets or {}).items():
                cls, grid = dn(xd)
                out[f"cls_{nm}"][i:i + b] = cls.cpu(); out[f"grid_{nm}"][i:i + b] = grid.half().cpu()
        if "clip" in want:
            out["clip"][i:i + b] = nets.clip.image_embed(_norm(c, CLIP_MEAN, CLIP_STD)).cpu()
        if "cnn" in views:
            xr = _norm(r, IM_MEAN, IM_STD)
            if "r50" in want:
                out["r50"][i:i + b] = nets.r50(xr).cpu()
            if "wrn" in want:
                out["wgrid"][i:i + b] = nets.wrn(xr).half().cpu()
        i += b
    return out

def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s, dtype=np.float64)
    if len(np.unique(y)) < 2 or not np.all(np.isfinite(s)):
        return np.nan
    return float(roc_auc_score(y, s))

def best_f1_threshold(y, s):
    y = np.asarray(y).astype(np.int64); s = np.asarray(s, dtype=np.float64)
    order = np.argsort(-s, kind="mergesort")
    ys, ss = y[order], s[order]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    distinct = np.r_[ss[1:] != ss[:-1], True]
    tp, fp, thr = tp[distinct], fp[distinct], ss[distinct]
    P = ys.sum()
    f1 = 2 * tp / np.maximum(tp + fp + P, 1)
    j = int(np.argmax(f1))
    return float(thr[j]), float(f1[j])

def bin_metrics(y, s, thr):
    y = np.asarray(y).astype(int); pred = (np.asarray(s) >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    n = max(len(y), 1)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(acc=(tp + tn) / n, prec=prec, rec=rec, f1=f1, fpr=fp / max(fp + tn, 1), tpr=rec,
                tp=tp, fp=fp, tn=tn, fn=fn)

def build_bank(grid, ref_idx, rng, ratio=BANK_RATIO, pool_max=BANK_POOL_MAX):
    P = grid.shape[1]
    pool = len(ref_idx) * P
    n_bank = max(1, int(min(pool, pool_max) * ratio))
    flat = rng.choice(pool, n_bank, replace=False)
    img = np.asarray(ref_idx)[flat // P]; pos = flat % P
    order = np.argsort(img, kind="mergesort"); img, pos = img[order], pos[order]
    parts = []
    for u in np.unique(img):
        sel = pos[img == u]
        parts.append(grid[int(u)][torch.from_numpy(sel)])
    return torch.cat(parts, 0).to(DEVICE).float()

def all_patch_bank(grid, idx):
    return grid[torch.as_tensor(np.asarray(idx))].reshape(-1, grid.shape[-1]).to(DEVICE).float()

@torch.no_grad()
def nn_maps(grid, idx, bank, bs=8):
    idx = np.asarray(idx); P = grid.shape[1]
    out = np.empty((len(idx), P), dtype=np.float32)
    bsq = (bank * bank).sum(1)[None, :]
    for i in range(0, len(idx), bs):
        sel = idx[i:i + bs]
        x = grid[torch.from_numpy(sel)].to(DEVICE).float().reshape(-1, grid.shape[-1])
        d2 = (x * x).sum(1, keepdim=True) + bsq - 2.0 * (x @ bank.T)
        out[i:i + len(sel)] = d2.min(1).values.clamp_min_(0).sqrt_().reshape(len(sel), P).cpu().numpy()
    return out

def agg_top3(maps):
    return np.sort(maps, axis=1)[:, -TOPK_IMG:].mean(1)

def agg_top1pct(maps):
    k = max(1, int(round(0.01 * maps.shape[1])))
    return np.sort(maps, axis=1)[:, -k:].mean(1)

NORMAL_STATES = ["flawless", "perfect", "unblemished", "normal"]
ANOMALY_STATES = ["damaged", "defective", "broken", "flawed"]
PROMPT_TEMPLATES = ["a photo of a {}.", "a close-up photo of a {}.", "a cropped photo of the {}.",
                    "a photo of the {} for visual inspection.", "a blurry photo of a {}."]

def clip_zero_shot_scores(clip_img, nets, ds, cat, dtypes_test, n_tr):
    cname = cat.replace("_", " ")
    t = nets.clip.text_embed([f"a photo of a normal {cname}", f"a photo of a {cname} with a defect"])
    z_generic = (clip_img @ t[1]) - (clip_img @ t[0])
    obj = object_name(ds, cat)
    def ens(states):
        ps = [tpl.format(f"{s} {obj}") for s in states for tpl in PROMPT_TEMPLATES]
        e = nets.clip.text_embed(ps).mean(0)
        return e / e.norm()
    z_ens = (clip_img @ ens(ANOMALY_STATES)) - (clip_img @ ens(NORMAL_STATES))
    out = dict(z_generic=z_generic.numpy(), z_ens=z_ens.numpy())
    classes = sorted(set(dtypes_test))
    prompts = [(f"a photo of a normal {cname} with no defect" if c == "good" else f"a photo of a {cname} with {c.replace('_', ' ')}")
               for c in classes]
    out["classes"] = classes
    out["sims_named"] = (clip_img[n_tr:] @ nets.clip.text_embed(prompts).T).numpy()
    return out

def softmax(v):
    v = np.asarray(v, dtype=np.float64); e = np.exp(v - v.max()); return e / e.sum()

L_RULES = ["uniform", "var_a1", "var_a10", "var_a50", "auc_w", "logreg", "sel_best", "max_z"]
N_RULES = ["uniform_z", "uniform_mm", "max_z"]

FUSION_CONFIGS = [
    ("orig", ("P",)), ("orig", ("Z",)), ("orig", ("A",)),
    ("orig", ("P", "Z")), ("orig", ("P", "A")), ("orig", ("Z", "A")), ("orig", ("P", "Z", "A")),
    ("prompt_ens", ("Zp",)), ("prompt_ens", ("P", "Zp")), ("prompt_ens", ("Zp", "A")), ("prompt_ens", ("P", "Zp", "A")),
    ("cnn_swap", ("Pc",)), ("cnn_swap", ("Aw",)), ("cnn_swap", ("Pc", "Z")), ("cnn_swap", ("Pc", "Aw")),
    ("cnn_swap", ("Z", "Aw")), ("cnn_swap", ("Pc", "Z", "Aw")),
]

def _row_base(ds, cat, seed, regime, policy, family, bs, rule):
    return dict(dataset=ds, category=cat, seed=seed, regime=regime, policy=policy, family=family,
                subset="+".join(bs), rule=rule)

def fuse_L(rule, Xc, yc, Xe, Xv):
    m = Xc.shape[1]
    lo, hi = Xc.min(0), Xc.max(0); rg = np.maximum(hi - lo, 1e-8)
    Nc, Ne = (Xc - lo) / rg, (Xe - lo) / rg
    w = np.ones(m) / m
    if rule == "uniform":
        return Nc @ w, Ne @ w, w
    if rule.startswith("var_a"):
        a = float(rule[5:]); w = softmax(a * Nc.var(0)); return Nc @ w, Ne @ w, w
    aucs = np.array([auroc(yc, Xc[:, j]) for j in range(m)])
    if rule == "auc_w":
        w = np.clip(aucs - 0.5, 0, None)
        w = np.ones(m) / m if w.sum() < 1e-9 else w / w.sum()
        return Nc @ w, Ne @ w, w
    if rule == "sel_best":
        j = int(np.nanargmax(aucs)); w = np.zeros(m); w[j] = 1.0
        return Nc[:, j], Ne[:, j], w
    if rule == "logreg":
        mu, sd = Xc.mean(0), Xc.std(0) + 1e-8
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
        clf.fit((Xc - mu) / sd, yc)
        return clf.decision_function((Xc - mu) / sd), clf.decision_function((Xe - mu) / sd), clf.coef_[0]
    if rule == "max_z":
        mu, sd = Xv.mean(0), Xv.std(0) + 1e-8
        return ((Xc - mu) / sd).max(1), ((Xe - mu) / sd).max(1), w
    raise ValueError(rule)

def eval_regime_L(ds, cat, seed, family, bs, Sv, St, sp, y):
    if len(sp["cal"]) == 0:
        return [], {}
    yc = np.r_[np.zeros(len(sp["val"])), np.ones(len(sp["cal"]))].astype(int)
    ye = y[sp["ev"]]
    Xv = np.stack([Sv[b] for b in bs], 1)
    Xc = np.concatenate([Xv, np.stack([St[b][sp["cal"]] for b in bs], 1)], 0)
    Xe = np.stack([St[b][sp["ev"]] for b in bs], 1)
    rules = ["single"] if len(bs) == 1 else L_RULES
    rows, scores = [], {}
    for rule in rules:
        r = "uniform" if rule == "single" else rule
        try:
            fc, fe, w = fuse_L(r, Xc, yc, Xe, Xv)
        except Exception as ex:
            continue
        thr, _ = best_f1_threshold(yc, fc)
        row = _row_base(ds, cat, seed, "L", "cal_f1max", family, bs, rule)
        row.update(auroc_eval=auroc(ye, fe), auroc_full=np.nan, thr=thr, **bin_metrics(ye, fe, thr))
        othr, _ = best_f1_threshold(ye, fe)
        ob = bin_metrics(ye, fe, othr)
        row.update(oracle_acc=ob["acc"], oracle_prec=ob["prec"], oracle_rec=ob["rec"], oracle_f1=ob["f1"],
                   oracle_fpr=ob["fpr"], w="|".join(f"{v:.4f}" for v in np.atleast_1d(w)))
        rows.append(row); scores[rule] = fe
    return rows, scores

def eval_regime_N(ds, cat, seed, family, bs, Sv, St, sp, y):
    Xv = np.stack([Sv[b] for b in bs], 1); Xt = np.stack([St[b] for b in bs], 1)
    mu, sd = Xv.mean(0), Xv.std(0) + 1e-8
    lo, hi = Xv.min(0), Xv.max(0); rg = np.maximum(hi - lo, 1e-8)
    rules = ["single"] if len(bs) == 1 else N_RULES
    rows = []
    for rule in rules:
        if rule in ("single", "uniform_z"):
            fv, ft = ((Xv - mu) / sd).mean(1), ((Xt - mu) / sd).mean(1)
        elif rule == "uniform_mm":
            fv, ft = ((Xv - lo) / rg).mean(1), ((Xt - lo) / rg).mean(1)
        else:
            fv, ft = ((Xv - mu) / sd).max(1), ((Xt - mu) / sd).max(1)
        a_full = auroc(y, ft); a_eval = auroc(y[sp["ev"]], ft[sp["ev"]])
        for q in THR_QUANTILES:
            thr = float(np.percentile(fv, q))
            row = _row_base(ds, cat, seed, "N", f"q{q}", family, bs, rule)
            row.update(auroc_eval=a_eval, auroc_full=a_full, thr=thr, **bin_metrics(y, ft, thr))
            rows.append(row)
    return rows

def upsample_maps(m, sigma=0.0):
    t = torch.from_numpy(np.asarray(m, dtype=np.float32)).reshape(-1, 1, GRID, GRID).to(DEVICE)
    up = F.interpolate(t, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    if sigma > 0:
        r = int(4 * sigma + 0.5)
        x = torch.arange(-r, r + 1, device=DEVICE, dtype=torch.float32)
        k = torch.exp(-0.5 * (x / sigma) ** 2); k = (k / k.sum())
        up = F.pad(up, (r, r, r, r), mode="reflect")
        up = F.conv2d(up, k.view(1, 1, 1, -1)); up = F.conv2d(up, k.view(1, 1, -1, 1))
    return up.squeeze(1)

def pixel_auroc_ap(scores, labels):
    n = scores.numel(); n_pos = int(labels.sum().item()); n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan, np.nan
    order = torch.argsort(scores)                       
    lab = labels[order].double()
    ranks = torch.arange(1, n + 1, device=scores.device, dtype=torch.float64)
    auc = ((ranks * lab).sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    lab_d = torch.flip(lab, [0])                        
    tp = torch.cumsum(lab_d, 0)
    prec = tp / torch.arange(1, n + 1, device=scores.device, dtype=torch.float64)
    ap = (prec * lab_d).sum() / n_pos
    return float(auc.item()), float(ap.item())

def aupro(maps_t, gts, fpr_max=0.3, n_pts=301, max_norm_px=4_000_000, seed=0):
    rng = np.random.RandomState(seed)
    comps = []; normal_chunks = []
    maps = maps_t.cpu().numpy()
    for i in range(len(gts)):
        g = gts[i]
        normal_chunks.append(maps[i][g == 0])
        if g.any():
            lab, nc = ndi.label(g)
            for c in range(1, nc + 1):
                comps.append(np.sort(maps[i][lab == c]))
    if not comps:
        return np.nan
    normals = np.concatenate(normal_chunks)
    if len(normals) > max_norm_px:
        normals = normals[rng.choice(len(normals), max_norm_px, replace=False)]
    fpr_grid = np.linspace(0.0, fpr_max, n_pts)
    thr = np.quantile(normals, 1.0 - fpr_grid)
    pro = np.zeros(n_pts)
    for cs in comps:
        pro += 1.0 - np.searchsorted(cs, thr, side="right") / len(cs)
    pro /= len(comps)
    area = np.sum((pro[1:] + pro[:-1]) * 0.5 * np.diff(fpr_grid))
    return float(area / fpr_max)

def region_counts(pred, gt, thrs=(0.1, 0.3)):
    lp, npc = ndi.label(pred); lg, ngc = ndi.label(gt)
    res = {t: [0, 0] for t in thrs}
    if npc > 0 and ngc > 0:
        pair = lg.astype(np.int64) * (npc + 1) + lp
        cnt = np.bincount(pair.ravel(), minlength=(ngc + 1) * (npc + 1)).reshape(ngc + 1, npc + 1)
        ag, ap_ = cnt.sum(1), cnt.sum(0)
        inter = cnt[1:, 1:]; union = ag[1:, None] + ap_[None, 1:] - inter
        iou = inter / np.maximum(union, 1)
        for t in thrs:
            res[t] = [int((iou.max(1) >= t).sum()), int((iou.max(0) >= t).sum())]
    return ngc, npc, res

def boundary_f(pred, gt, tol=3):
    def bd(m):
        return m & ~ndi.binary_erosion(m, iterations=1, border_value=0)
    bp, bg = bd(pred), bd(gt)
    if bg.sum() == 0:
        return np.nan
    if bp.sum() == 0:
        return 0.0
    dg = ndi.distance_transform_edt(~bg); dp = ndi.distance_transform_edt(~bp)
    prec = float((dg[bp] <= tol).mean()); rec = float((dp[bg] <= tol).mean())
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)

def localization_eval(maps_test, maps_val, gt, legacy_gt, keep, sigma):
    idx = np.where(keep)[0]
    up = upsample_maps(maps_test[idx], sigma)
    g = gt[idx]; lg = legacy_gt[idx]
    flat = up.reshape(-1)
    auc, ap = pixel_auroc_ap(flat, torch.from_numpy(g.reshape(-1)).to(DEVICE))
    auc_legacy, _ = pixel_auroc_ap(flat, torch.from_numpy(lg.reshape(-1)).to(DEVICE))
    pro = aupro(up, g)
    upv = upsample_maps(maps_val, sigma).reshape(-1).cpu().numpy()
    tau = float(np.percentile(upv, PIX_Q)) if len(upv) < 30_000_000 else float(np.percentile(upv[::3], PIX_Q))
    pred_all = (up > tau).cpu().numpy()
    tp = fp = fn = 0; ngt = npd = 0; hit = {0.1: [0, 0], 0.3: [0, 0]}; bfs = []
    for k in range(len(idx)):
        pr, gg = pred_all[k], g[k].astype(bool)
        tp += int((pr & gg).sum()); fp += int((pr & ~gg).sum()); fn += int((~pr & gg).sum())
        a, b, res = region_counts(pr, gg)
        ngt += a; npd += b
        for t in res:
            hit[t][0] += res[t][0]; hit[t][1] += res[t][1]
        if gg.any():
            bfs.append(boundary_f(pr, gg))
    out = dict(pixel_auroc=auc * 100, pixel_auroc_legacy_mask=auc_legacy * 100, pixel_ap=ap * 100, aupro03=pro * 100,
               pix_thr=tau, dice=100 * 2 * tp / max(2 * tp + fp + fn, 1), iou=100 * tp / max(tp + fp + fn, 1),
               pix_prec=100 * tp / max(tp + fp, 1), pix_rec=100 * tp / max(tp + fn, 1),
               boundary_f=100 * float(np.nanmean(bfs)) if len(bfs) else np.nan, n_gt_regions=ngt, n_pred_regions=npd)
    for t in (0.1, 0.3):
        p = hit[t][1] / max(npd, 1); r = hit[t][0] / max(ngt, 1)
        out[f"region_prec_iou{t}"] = 100 * p; out[f"region_rec_iou{t}"] = 100 * r
        out[f"region_f1_iou{t}"] = 100 * (2 * p * r / (p + r) if p + r else 0.0)
    del up, flat
    return out

class CatData:
    pass

def split_for_seed(cd, seed):
    rng = np.random.RandomState(SEED_BASE + seed)
    perm = rng.permutation(cd.n_tr)
    n_val = int(max(MIN_VAL, round(VAL_FRAC * cd.n_tr))); n_val = min(n_val, cd.n_tr // 2)
    val = np.sort(perm[:n_val]); ref = np.sort(perm[n_val:])
    rng2 = np.random.RandomState(SEED_BASE + 500 + seed)
    d_idx = np.where(cd.y == 1)[0]; n_def = len(d_idx)
    n_cal = max(MIN_CAL_DEF, int(round(CAL_DEF_FRAC * n_def)))
    if n_def - n_cal < MIN_CAL_DEF:
        n_cal = 0
    cal = np.sort(rng2.choice(d_idx, n_cal, replace=False)) if n_cal else np.array([], dtype=int)
    ev = np.setdiff1d(np.arange(cd.n_te), cal)
    return dict(ref=ref, val=val, cal=cal, ev=ev)

def _l2n(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

def compute_branches(cd, sp, seed):
    n_tr = cd.n_tr; te = np.arange(n_tr, n_tr + cd.n_te); val, ref = sp["val"], sp["ref"]
    Sv, St, M = {}, {}, {}
    c = cd.cls_n[ref].mean(0)
    Sv["P"] = np.linalg.norm(cd.cls_n[val] - c, axis=1); St["P"] = np.linalg.norm(cd.cls_n[te] - c, axis=1)
    cr = cd.r50[ref].mean(0)
    Sv["Pc"] = np.linalg.norm(cd.r50[val] - cr, axis=1); St["Pc"] = np.linalg.norm(cd.r50[te] - cr, axis=1)
    Sv["Z"], St["Z"] = cd.zs["z_generic"][val], cd.zs["z_generic"][te]
    Sv["Zp"], St["Zp"] = cd.zs["z_ens"][val], cd.zs["z_ens"][te]
    bank = build_bank(cd.grid, ref, np.random.RandomState(SEED_BASE + 900 + seed))
    mv, mt = nn_maps(cd.grid, val, bank), nn_maps(cd.grid, te, bank)
    Sv["A"], St["A"] = agg_top3(mv), agg_top3(mt); M["A"] = (mv, mt)
    bankw = build_bank(cd.wgrid, ref, np.random.RandomState(SEED_BASE + 950 + seed))
    wv, wt = nn_maps(cd.wgrid, val, bankw), nn_maps(cd.wgrid, te, bankw)
    Sv["Aw"], St["Aw"] = agg_top3(wv), agg_top3(wt); M["Aw"] = (wv, wt)
    del bankw
    return Sv, St, M, bank, c

def zero_shot_rows(cd):
    zs, y, dt = cd.zs, cd.y, cd.dtypes
    classes = zs["classes"]; yc = np.array([classes.index(c) for c in dt]); sims = zs["sims_named"]
    ranked = np.argsort(-sims, axis=1)
    top1 = float((ranked[:, 0] == yc).mean()); k5 = min(5, len(classes))
    top5 = float(np.mean([yc[i] in ranked[i, :k5] for i in range(len(yc))]))
    counts = np.bincount(yc, minlength=len(classes)); majority = float(counts.max() / len(yc))
    row = dict(dataset=cd.ds, category=cd.cat, n_classes=len(classes), top1=100 * top1, top5=100 * top5,
               chance=100.0 / len(classes), majority_class=100 * majority, defect_only_acc=np.nan, defect_only_chance=np.nan)
    if "good" in classes and len(classes) > 2:
        gi = classes.index("good"); dcols = [i for i in range(len(classes)) if i != gi]
        m = y == 1
        pred = np.array(dcols)[np.argmax(sims[m][:, dcols], axis=1)]
        row["defect_only_acc"] = 100 * float((pred == yc[m]).mean()); row["defect_only_chance"] = 100.0 / len(dcols)
    n_tr = cd.n_tr
    row["auroc_generic"] = 100 * auroc(y, zs["z_generic"][n_tr:])
    row["auroc_ensemble"] = 100 * auroc(y, zs["z_ens"][n_tr:])
    if "good" in classes:
        gi = classes.index("good"); dcols = [i for i in range(len(classes)) if i != gi]
        row["auroc_oracle_names"] = 100 * auroc(y, sims[:, dcols].max(1) - sims[:, gi])
    else:
        row["auroc_oracle_names"] = np.nan
    return row

def fewshot_rows(cd, sp, seed):
    rows = []; n_tr = cd.n_tr; y = cd.y
    d_idx = np.where(y == 1)[0]
    for k in FEWSHOT_KS:
        if len(d_idx) <= k + 1 or len(sp["ref"]) < k:
            continue
        rng = np.random.RandomState(SEED_BASE + 2000 + seed * 10 + k)
        sup_n = rng.choice(sp["ref"], k, replace=False); sup_d = rng.choice(d_idx, k, replace=False)
        q = np.setdiff1d(np.arange(cd.n_te), sup_d)
        pn = cd.cls_n[sup_n].mean(0); pdv = cd.cls_n[n_tr + sup_d].mean(0)
        Q = cd.cls_n[n_tr + q]
        s = np.linalg.norm(Q - pn, axis=1) - np.linalg.norm(Q - pdv, axis=1)      
        nat = bin_metrics(y[q], s, 0.0)                                           
        othr, _ = best_f1_threshold(y[q], s); orc = bin_metrics(y[q], s, othr)
        rows.append(dict(dataset=cd.ds, category=cd.cat, seed=seed, k=k,
                         auroc=100 * auroc(y[q], s), acc=100 * nat["acc"], f1=100 * nat["f1"], prec=100 * nat["prec"],
                         rec=100 * nat["rec"], oracle_acc=100 * orc["acc"], oracle_f1=100 * orc["f1"]))
    return rows

def normalshot_rows(cd, sp, seed):
    rows = []; n_tr = cd.n_tr; te = np.arange(n_tr, n_tr + cd.n_te)
    for k in NORMAL_SHOT_KS:
        if len(sp["ref"]) < k:
            continue
        rng = np.random.RandomState(SEED_BASE + 3000 + seed * 10 + k)
        sup = rng.choice(sp["ref"], k, replace=False)
        c = cd.cls_n[sup].mean(0)
        a_proto = auroc(cd.y, np.linalg.norm(cd.cls_n[te] - c, axis=1))
        bank = all_patch_bank(cd.grid, sup)
        mt = nn_maps(cd.grid, te, bank)
        rows.append(dict(dataset=cd.ds, category=cd.cat, seed=seed, k=k, bank_size=int(bank.shape[0]),
                         auroc_proto_normal=100 * a_proto, auroc_bank_top3=100 * auroc(cd.y, agg_top3(mt)),
                         auroc_bank_top1pct=100 * auroc(cd.y, agg_top1pct(mt))))
        del bank
    return rows

def bank_size_rows(cd, sp, seed):
    rows = []; n_tr = cd.n_tr; te = np.arange(n_tr, n_tr + cd.n_te)
    for ratio in BANK_RATIOS:
        bank = build_bank(cd.grid, sp["ref"], np.random.RandomState(SEED_BASE + 700 + seed), ratio=ratio)
        if DEVICE.type == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter(); mt = nn_maps(cd.grid, te, bank)
        if DEVICE.type == "cuda": torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / max(len(te), 1) * 1000
        rows.append(dict(dataset=cd.ds, category=cd.cat, seed=seed, ratio=ratio, bank_size=int(bank.shape[0]),
                         auroc=100 * auroc(cd.y, agg_top3(mt)), ms_per_image=ms))
        del bank
    return rows

def load_gts(cd):
    n = cd.n_te
    keep = np.ones(n, bool); gt = np.zeros((n, IMG_SIZE, IMG_SIZE), np.uint8); lgt = np.zeros_like(gt); n_m = 0
    for i in range(n):
        if cd.y[i] == 1:
            mp = cd.mask_p[i]
            if mp is None or not os.path.exists(mp):
                keep[i] = False; continue
            try:
                gt[i] = load_gt_mask(mp, "aligned"); lgt[i] = load_gt_mask(mp, "legacy"); n_m += 1
            except Exception:
                keep[i] = False
    if n_m == 0 or not gt.any():
        return None
    return gt, lgt, keep

def pick_examples(cd, mt, keep, mv_tau=None):
    s = agg_top3(mt); ex = []
    d = [i for i in np.where((cd.y == 1) & keep)[0]]; nrm = list(np.where(cd.y == 0)[0])
    if d:
        ex.append(("Highest-scoring defect", int(d[int(np.argmax(s[d]))])))
        ex.append(("Lowest-scoring defect", int(d[int(np.argmin(s[d]))])))
    if nrm:
        ex.append(("Highest-scoring normal", int(nrm[int(np.argmax(s[nrm]))])))
    return [dict(kind=k, path=cd.test_p[i], mask=cd.mask_p[i], y=int(cd.y[i]), score=float(s[i]), map=mt[i].copy()) for k, i in ex]

def process_category(task, nets, scale_nets):
    ds, cat, train_p, test_p, y_list, mask_p, dtypes = task
    t_start = time.perf_counter()
    y = np.array(y_list).astype(int)
    rng0 = np.random.RandomState(12345)
    cap = 24 if QUICK_TEST else TRAIN_CAP
    if QUICK_TEST:
        keep_te = np.r_[np.where(y == 0)[0][:12], np.where(y == 1)[0][:12]]; keep_te.sort()
        test_p = [test_p[i] for i in keep_te]; mask_p = [mask_p[i] for i in keep_te]
        dtypes = [dtypes[i] for i in keep_te]; y = y[keep_te]
    try:
        import psutil
        budget = 0.40 * psutil.virtual_memory().available
        max_imgs = int(budget / (NP_DINO * 1024 * 2 + 784 * 1024 * 2 + 20000))
        cap = int(max(60, min(cap, max_imgs - len(test_p))))
    except Exception:
        pass
    if len(train_p) > cap:
        sel = np.sort(rng0.choice(len(train_p), cap, replace=False)); train_p = [train_p[i] for i in sel]
    cd = CatData()
    cd.ds, cd.cat, cd.y, cd.dtypes, cd.mask_p, cd.test_p = ds, cat, y, list(dtypes), list(mask_p), list(test_p)
    cd.n_tr, cd.n_te = len(train_p), len(test_p)
    log(f"  {ds}/{cat}: {cd.n_tr} normal train (capped), {cd.n_te} test ({int((y == 0).sum())} normal / {int((y == 1).sum())} defective)")

    Fe = extract_features(train_p + test_p, nets, desc=f"{cat}-features")
    cd.cls_raw = Fe["cls"].numpy(); cd.cls_n = _l2n(cd.cls_raw)
    cd.grid, cd.wgrid, cd.r50, cd.clip = Fe["grid"], Fe["wgrid"], Fe["r50"].numpy(), Fe["clip"]
    cd.zs = clip_zero_shot_scores(cd.clip, nets, ds, cat, cd.dtypes, cd.n_tr)
    cname = cat.replace("_", " ")
    cd.t_generic = nets.clip.text_embed([f"a photo of a normal {cname}", f"a photo of a {cname} with a defect"])

    res = defaultdict(list)
    res["zeroshot"].append(zero_shot_rows(cd))
    gts = load_gts(cd) if ds not in ("crack",) else None
    if gts is None and cd.y.sum() > 0 and ds not in ("crack",):
        log(f"    [warn] no usable ground-truth masks for {ds}/{cat}: localisation skipped")
    rob_state = {}
    curves_cfg = {("orig", ("A",), "single"), ("orig", ("P", "Z", "A"), "uniform"), ("orig", ("P", "Z", "A"), "var_a1")}

    for seed in range(N_SEEDS):
        sp = split_for_seed(cd, seed)
        Sv, St, M, bank, cP = compute_branches(cd, sp, seed)
        if seed == 0:
            res["split"].append(dict(dataset=ds, category=cat, n_ref=len(sp["ref"]), n_val=len(sp["val"]), n_cal_defects=len(sp["cal"]),
                                     n_eval_normal=int((y[sp["ev"]] == 0).sum()), n_eval_defect=int((y[sp["ev"]] == 1).sum()),
                                     n_test_normal=int((y == 0).sum()), n_test_defect=int((y == 1).sum())))
        for fam, bs in FUSION_CONFIGS:
            rowsL, scores = eval_regime_L(ds, cat, seed, fam, bs, Sv, St, sp, y)
            res["fusion"].extend(rowsL)
            res["fusion"].extend(eval_regime_N(ds, cat, seed, fam, bs, Sv, St, sp, y))
            if seed == 0:
                for rule, fe in scores.items():
                    if (fam, bs, rule) in curves_cfg:
                        res["curves"].append(dict(dataset=ds, category=cat, key=f"{'+'.join(bs)}|{rule}", y=y[sp["ev"]].copy(), s=np.asarray(fe, dtype=np.float32)))
        if len(sp["cal"]):
            yc = np.r_[np.zeros(len(sp["val"])), np.ones(len(sp["cal"]))].astype(int)
            for b in ("P", "Z", "A"):
                xc = np.r_[Sv[b], St[b][sp["cal"]]]; nx = (xc - xc.min()) / max(xc.max() - xc.min(), 1e-8)
                res["variance"].append(dict(dataset=ds, category=cat, seed=seed, branch=b, var_cal=float(nx.var()),
                                            auc_cal=auroc(yc, xc), auc_eval=auroc(y[sp["ev"]], St[b][sp["ev"]])))
        res["fewshot"].extend(fewshot_rows(cd, sp, seed))
        res["normalshot"].extend(normalshot_rows(cd, sp, seed))
        if seed < BANK_SEEDS:
            res["bank"].extend(bank_size_rows(cd, sp, seed))
        if ds in ROBUSTNESS_DATASETS and seed < ROB_SEEDS and len(sp["cal"]):
            yc = np.r_[np.zeros(len(sp["val"])), np.ones(len(sp["cal"]))]
            lo, hi, var = {}, {}, {}
            for b in ("P", "Z", "A"):
                xc = np.r_[Sv[b], St[b][sp["cal"]]]; lo[b], hi[b] = xc.min(), max(xc.max(), xc.min() + 1e-8)
                var[b] = float(((xc - lo[b]) / (hi[b] - lo[b])).var())
            rob_state[seed] = dict(sp=sp, bank=bank, cP=cP, lo=lo, hi=hi, w1=softmax([var[b] for b in ("P", "Z", "A")]))
        if gts is not None and seed < LOC_SEEDS:
            gt, lgt, keep = gts
            mv, mt = M["A"]
            for sig_name, sig in (("raw", 0.0), ("gauss4", SMOOTH_SIGMA)):
                lm = localization_eval(mt, mv, gt, lgt, keep, sig)
                res["loc"].append(dict(dataset=ds, category=cat, seed=seed, smoothing=sig_name, n_images=int(keep.sum()), **lm))
                if seed == 0 and sig_name == "gauss4":
                    tau4 = lm["pix_thr"]
            if seed == 0:
                res["qual"] = pick_examples(cd, mt, keep); res["qual_meta"].append(dict(tau=tau4))
        del bank
        cleanup()
    if rob_state:
        res["robust"].extend(run_robustness(cd, nets, rob_state))
    if ds in BACKBONE_SCALING_DATASETS and scale_nets:
        res["scaling"].extend(run_scaling(cd, nets, scale_nets, train_p))
    res["timing"].append(dict(dataset=ds, category=cat, seconds=time.perf_counter() - t_start))
    del cd, Fe
    cleanup()
    return dict(res)

def run_robustness(cd, nets, rob_state):
    rows = []; te_rows = np.arange(cd.n_tr, cd.n_tr + cd.n_te)
    t_gen = cd.t_generic
    for corr in ["clean"] + ROB_CORRUPTIONS:
        if corr == "clean":
            cls, grid, clipf = cd.cls_raw[te_rows], cd.grid[cd.n_tr:], cd.clip[cd.n_tr:]
        else:
            Fc = extract_features(cd.test_p, nets, want=("dino", "clip"), corruption=corr, desc=f"{cd.cat}-{corr}")
            cls, grid, clipf = Fc["cls"].numpy(), Fc["grid"], Fc["clip"]
        cls_n = _l2n(cls); zc = (clipf @ t_gen[1] - clipf @ t_gen[0]).numpy()
        for seed, st in rob_state.items():
            sc = {"P": np.linalg.norm(cls_n - st["cP"], axis=1), "Z": zc,
                  "A": agg_top3(nn_maps(grid, np.arange(cd.n_te), st["bank"]))}
            nz = {b: (sc[b] - st["lo"][b]) / (st["hi"][b] - st["lo"][b]) for b in sc}
            ev = st["sp"]["ev"]
            fused_u = (nz["P"] + nz["Z"] + nz["A"]) / 3
            fused_v = st["w1"][0] * nz["P"] + st["w1"][1] * nz["Z"] + st["w1"][2] * nz["A"]
            cand = dict(P=sc["P"], Z=sc["Z"], A=sc["A"], HFM_uniform=fused_u, HFM_adaptive=fused_v)
            for name, v in cand.items():
                rows.append(dict(dataset=cd.ds, category=cd.cat, seed=seed, corruption=corr, method=name,
                                 auroc=100 * auroc(cd.y[ev], v[ev])))
        del grid
    return rows

def run_scaling(cd, nets, scale_nets, train_used):
    rows = []
    allp = list(train_used) + list(cd.test_p)
    Fs = extract_features(allp, nets, want=(), dino_nets=scale_nets, desc=f"{cd.cat}-scaling")
    te = np.arange(cd.n_tr, cd.n_tr + cd.n_te)
    for seed in range(BANK_SEEDS):
        sp = split_for_seed(cd, seed)
        for nm in scale_nets:
            grid, cls = Fs[f"grid_{nm}"], _l2n(Fs[f"cls_{nm}"].numpy())
            bank = build_bank(grid, sp["ref"], np.random.RandomState(SEED_BASE + 900 + seed))
            a = auroc(cd.y, agg_top3(nn_maps(grid, te, bank)))
            p = auroc(cd.y, np.linalg.norm(cls[te] - cls[sp["ref"]].mean(0), axis=1))
            rows.append(dict(dataset=cd.ds, category=cd.cat, seed=seed, backbone=nm, A_auroc=100 * a, P_auroc=100 * p))
            del bank
    del Fs
    return rows

def count_params(m):
    return sum(p.numel() for p in m.parameters())

def vit_macs(n_tokens, dim, depth, patch_in=None, n_patches=None):
    per_layer = 12 * n_tokens * dim * dim + 2 * n_tokens * n_tokens * dim
    emb = (n_patches * patch_in * dim) if n_patches else 0
    return depth * per_layer + emb

def flops_of(fn, *args):
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(display=False) as fc:
            fn(*args)
        return fc.get_total_flops()
    except Exception as e:
        log("   [warn] FlopCounterMode failed:", e); return float("nan")

def _sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

def time_fn(fn, n_warm=20, n_run=100):
    for _ in range(n_warm): fn()
    _sync(); ts = []
    for _ in range(n_run):
        _sync(); t0 = time.perf_counter(); fn(); _sync(); ts.append((time.perf_counter() - t0) * 1000)
    ts = np.array(ts)
    return dict(mean_ms=float(ts.mean()), std_ms=float(ts.std()), p50_ms=float(np.percentile(ts, 50)), p95_ms=float(np.percentile(ts, 95)))

@torch.no_grad()
def run_efficiency(nets, scale_nets, example_image_path):
    log("\n" + "=" * 80 + "\nEFFICIENCY BENCHMARK\n" + "=" * 80)
    dino, clip = nets.dino, nets.clip
    p_dino = count_params(dino.model)
    p_clip_v = count_params(clip.model.vision_model) + count_params(clip.model.visual_projection)
    p_clip_t = count_params(clip.model.text_model) + count_params(clip.model.text_projection)
    p_r50, p_wrn = count_params(nets.r50.model), count_params(nets.wrn.model)
    x336 = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE); x224 = torch.randn(1, 3, 224, 224, device=DEVICE)
    f_dino = flops_of(lambda: dino(x336))
    f_clip = flops_of(lambda: clip.image_embed(x224))
    f_r50 = flops_of(lambda: nets.r50(x224)); f_wrn = flops_of(lambda: nets.wrn(x224))
    m_dino_an = vit_macs(577, 1024, 24, 14 * 14 * 3, 576); m_clip_an = vit_macs(257, 1024, 24, 14 * 14 * 3, 256)
    bank_macs = NP_DINO * int(BANK_POOL_MAX * BANK_RATIO) * 1024
    rows = [
        dict(component="DINOv2 ViT-L/14 (336x336, cls + 4-layer patch grid)", params_M=p_dino / 1e6, GMACs_measured=f_dino / 2e9, GMACs_analytic=m_dino_an / 1e9, used_at_inference=True),
        dict(component="CLIP ViT-L/14 image tower + projection (224x224)", params_M=p_clip_v / 1e6, GMACs_measured=f_clip / 2e9, GMACs_analytic=m_clip_an / 1e9, used_at_inference=True),
        dict(component="CLIP text tower + projection (prompts embedded once, offline)", params_M=p_clip_t / 1e6, GMACs_measured=np.nan, GMACs_analytic=np.nan, used_at_inference=False),
        dict(component="Nearest-neighbour search (576 patches x 6000 bank x 1024-D)", params_M=0.0, GMACs_measured=bank_macs / 1e9, GMACs_analytic=bank_macs / 1e9, used_at_inference=True),
        dict(component="ResNet-50 (224x224)", params_M=p_r50 / 1e6, GMACs_measured=f_r50 / 2e9, GMACs_analytic=np.nan, used_at_inference=False),
        dict(component="Wide-ResNet-50-2 PatchCore features (224x224)", params_M=p_wrn / 1e6, GMACs_measured=f_wrn / 2e9, GMACs_analytic=np.nan, used_at_inference=False),
    ]
    for nm, dn in (scale_nets or {}).items():
        d = dn.dim; L = 12
        rows.append(dict(component=f"DINOv2 {nm} (336x336)", params_M=count_params(dn.model) / 1e6,
                         GMACs_measured=flops_of(lambda: dn(x336)) / 2e9, GMACs_analytic=vit_macs(577, d, L, 588, 576) / 1e9, used_at_inference=False))
    df = pd.DataFrame(rows)
    inf_params = (p_dino + p_clip_v) / 1e6
    summ = pd.DataFrame([dict(
        item="Stored weights, DINOv2-L + complete CLIP-L (as in the submitted version, without the unused MLP head)", value=(p_dino + p_clip_v + p_clip_t) / 1e6, unit="M parameters"),
        dict(item="Inference path, HFM (DINOv2-L + CLIP-L image tower)", value=inf_params, unit="M parameters"),
        dict(item="Inference path, memory-bank branch only (DINOv2-L)", value=p_dino / 1e6, unit="M parameters"),
        dict(item="HFM total compute incl. attention (measured)", value=(f_dino + f_clip) / 2e9 + bank_macs / 1e9, unit="GMACs"),
        dict(item="HFM total compute incl. attention (analytic)", value=(m_dino_an + m_clip_an + bank_macs) / 1e9, unit="GMACs")])
    save_table(df, "efficiency_params_flops.csv"); save_table(summ, "efficiency_summary.csv")
    log(df.round(2).to_markdown(index=False)); log(summ.round(2).to_markdown(index=False))

    bank = torch.randn(int(BANK_POOL_MAX * BANK_RATIO), 1024, device=DEVICE); bsq = (bank * bank).sum(1)[None, :]
    t_prompt = clip.text_embed(["a photo of a normal object", "a photo of an object with a defect"]).to(DEVICE)
    ds_img = MultiViewDataset([example_image_path])
    def stage_pre(): ds_img[0]
    d_u8, c_u8, r_u8 = ds_img[0]
    d_u8, c_u8, r_u8 = d_u8.unsqueeze(0), c_u8.unsqueeze(0), r_u8.unsqueeze(0)
    def stage_h2d(): _norm(d_u8, IM_MEAN, IM_STD); _norm(c_u8, CLIP_MEAN, CLIP_STD)
    xd, xc = _norm(d_u8, IM_MEAN, IM_STD), _norm(c_u8, CLIP_MEAN, CLIP_STD)
    def stage_dino(): dino(xd)
    def stage_clip(): clip.image_embed(xc)
    cls0, grid0 = dino(xd)
    def nn_stage(g):
        x = g.reshape(-1, 1024).float(); d2 = (x * x).sum(1, keepdim=True) + bsq - 2.0 * (x @ bank.T)
        return d2.min(1).values.clamp_min(0).sqrt().topk(TOPK_IMG).values.mean()
    def stage_nn(): nn_stage(grid0)
    a0 = nn_stage(grid0); p0 = (cls0 - cls0.mean()).norm(); z0 = (clip.image_embed(xc) @ t_prompt.T)
    def stage_fuse():
        s = (a0 + p0 + z0[0, 1] - z0[0, 0]) / 3; bool(s > 0.5)
    def e2e_full():
        d, c, r = ds_img[0]; xd_ = _norm(d.unsqueeze(0), IM_MEAN, IM_STD); xc_ = _norm(c.unsqueeze(0), CLIP_MEAN, CLIP_STD)
        cl, gr = dino(xd_); z = clip.image_embed(xc_) @ t_prompt.T
        a = nn_stage(gr); p = (cl - cl.mean()).norm(); s = (a + p + z[0, 1] - z[0, 0]) / 3; bool(s > 0.5)
    def e2e_bank_only():
        d, c, r = ds_img[0]; xd_ = _norm(d.unsqueeze(0), IM_MEAN, IM_STD); cl, gr = dino(xd_); bool(nn_stage(gr) > 0.5)
    lat = []
    for name, fn in [("1 decode + 3 resized views (CPU)", stage_pre), ("2 normalisation + host-to-device copy", stage_h2d),
                     ("3 DINOv2-L forward (cls + patch grid)", stage_dino), ("4 CLIP-L image tower forward", stage_clip),
                     ("5 nearest-neighbour search + top-3 score", stage_nn), ("6 score normalisation, fusion and threshold decision", stage_fuse),
                     ("END-TO-END, full HFM (stages 1-6, one image)", e2e_full), ("END-TO-END, memory-bank branch only (DINOv2-L)", e2e_bank_only)]:
        if DEVICE.type == "cuda": torch.cuda.reset_peak_memory_stats()
        r = time_fn(fn, 10, 100); r["stage"] = name
        r["peak_gpu_MB"] = torch.cuda.max_memory_allocated() / 2**20 if DEVICE.type == "cuda" else np.nan
        lat.append(r)
    def mk(fn_, x): return lambda: fn_(x)
    alt = [("DINOv2-L/14 @336 (bank branch)", mk(dino, x336)), ("CLIP-L/14 image tower @224", mk(clip.image_embed, x224)),
           ("ResNet-50 global @224", mk(nets.r50, x224)), ("Wide-ResNet-50-2 PatchCore @224", mk(nets.wrn, x224))]
    for nm, dn in (scale_nets or {}).items():
        alt.append((f"DINOv2 {nm} @336 (bank branch)", mk(dn, x336)))
    alat = []
    for name, fn in alt:
        if DEVICE.type == "cuda": torch.cuda.reset_peak_memory_stats()
        r = time_fn(fn, 10, 100); r["model"] = name
        r["peak_gpu_MB"] = torch.cuda.max_memory_allocated() / 2**20 if DEVICE.type == "cuda" else np.nan
        alat.append(r)
    dl, da = pd.DataFrame(lat), pd.DataFrame(alat)
    save_table(dl, "efficiency_latency_breakdown.csv"); save_table(da, "efficiency_latency_backbones.csv")
    log(dl.round(2).to_markdown(index=False)); log(da.round(2).to_markdown(index=False))
    thr = []
    for bs in (1, 4, 8, 16, 32):
        try:
            xb = torch.randn(bs, 3, IMG_SIZE, IMG_SIZE, device=DEVICE); xcb = torch.randn(bs, 3, 224, 224, device=DEVICE)
            fn = lambda: (dino(xb), clip.image_embed(xcb))
            if DEVICE.type == "cuda": torch.cuda.reset_peak_memory_stats()
            r = time_fn(fn, 3, 15)
            thr.append(dict(batch=bs, ms_per_batch=r["mean_ms"], ms_per_image=r["mean_ms"] / bs, images_per_s=1000 * bs / r["mean_ms"],
                            peak_gpu_MB=torch.cuda.max_memory_allocated() / 2**20 if DEVICE.type == "cuda" else np.nan))
        except Exception as e:
            thr.append(dict(batch=bs, ms_per_batch=np.nan, note=str(e)[:80])); cleanup(); break
    dt = pd.DataFrame(thr); save_table(dt, "efficiency_throughput_batching.csv"); log(dt.round(2).to_markdown(index=False))
    del bank; cleanup()
    return dict(components=df, summary=summ, latency=dl, backbones_latency=da, throughput=dt)

def with_all(df):
    if df is None or len(df) == 0:
        return df
    a = df.copy(); a["category"] = a["dataset"] + "/" + a["category"]; a["dataset"] = "ALL"
    return pd.concat([df, a], ignore_index=True)

def _boot_ci(v, B=2000, seed=0):
    v = np.asarray(v, dtype=np.float64); v = v[np.isfinite(v)]
    if len(v) < 2:
        return np.nan, np.nan
    rng = np.random.RandomState(seed)
    m = v[rng.randint(0, len(v), size=(B, len(v)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

def summarize(df, keys, vals, boot_cols=(), pooled=True):

    if df is None or len(df) == 0:
        return pd.DataFrame()
    d = with_all(df) if pooled else df
    res = None
    for v in vals:
        dv = d.dropna(subset=[v])
        if len(dv) == 0:
            continue
        gc_ = dv.groupby(["dataset", "category"] + keys)[v].mean().reset_index()
        gs_ = dv.groupby(["dataset", "seed"] + keys)[v].mean().reset_index()
        mean = gc_.groupby(["dataset"] + keys)[v].mean().rename(v)
        sd = gs_.groupby(["dataset"] + keys)[v].std(ddof=1).rename(v + "_std")
        ns = gs_.groupby(["dataset"] + keys)[v].size().rename("n_seeds")
        nc = gc_.groupby(["dataset"] + keys)[v].size().rename("n_categories")
        part = pd.concat([mean, sd, ns, nc], axis=1)
        tq = sps.t.ppf(0.975, np.maximum(part["n_seeds"] - 1, 1))
        part[v + "_ci_seed"] = tq * part[v + "_std"] / np.sqrt(part["n_seeds"])
        if v in boot_cols:
            ci = gc_.groupby(["dataset"] + keys)[v].apply(lambda x: _boot_ci(x.values))
            part[v + "_ci_cat_lo"] = [c[0] for c in ci.reindex(part.index)]
            part[v + "_ci_cat_hi"] = [c[1] for c in ci.reindex(part.index)]
        if res is not None:
            part = part.drop(columns=["n_seeds", "n_categories"])
        res = part if res is None else res.join(part, how="outer")
    return res.reset_index() if res is not None else pd.DataFrame()

def holm(p):
    p = np.asarray(p, dtype=np.float64); n = len(p); order = np.argsort(p); adj = np.empty(n); run = 0.0
    for r, i in enumerate(order):
        run = max(run, (n - r) * p[i]); adj[i] = min(run, 1.0)
    return adj

def paired_contrast(fus, regime, policy, cfgA, cfgB, metric="auroc_eval", label=None):
    def pick(cfg):
        f, s, r = cfg
        d = fus[(fus.regime == regime) & (fus.policy == policy) & (fus.family == f) & (fus.subset == s) & (fus.rule == r)]
        return d[["dataset", "category", "seed", metric]]
    a, b = pick(cfgA), pick(cfgB)
    m = a.merge(b, on=["dataset", "category", "seed"], suffixes=("_a", "_b")).dropna()
    if len(m) == 0:
        return []
    m["delta"] = 100 * (m[metric + "_a"] - m[metric + "_b"])
    rows = []
    for ds, g in list(m.groupby("dataset")) + [("ALL", m.assign(category=m.dataset + "/" + m.category))]:
        pc = g.groupby("category")["delta"].mean()
        ps = g.groupby("seed")["delta"].mean()
        lo, hi = _boot_ci(pc.values, B=BOOT_N)
        try:
            p_w = float(sps.wilcoxon(pc.values).pvalue) if (pc.values != 0).any() and len(pc) >= 5 else np.nan
        except Exception:
            p_w = np.nan
        tq = sps.t.ppf(0.975, max(len(ps) - 1, 1))
        rows.append(dict(contrast=label, dataset=ds, regime=regime, policy=policy, metric=metric, n_categories=len(pc), n_seeds=len(ps),
                         mean_delta_pp=float(pc.mean()), sd_over_categories=float(pc.std(ddof=1)) if len(pc) > 1 else np.nan,
                         ci_cat_lo=lo, ci_cat_hi=hi, ci_seed_lo=float(ps.mean() - tq * ps.std(ddof=1) / np.sqrt(len(ps))) if len(ps) > 1 else np.nan,
                         ci_seed_hi=float(ps.mean() + tq * ps.std(ddof=1) / np.sqrt(len(ps))) if len(ps) > 1 else np.nan,
                         wilcoxon_p=p_w, frac_cat_seed_pos=float((g["delta"] > 0).mean()), frac_cat_pos=float((pc > 0).mean()),
                         cohens_dz=float(pc.mean() / pc.std(ddof=1)) if len(pc) > 1 and pc.std(ddof=1) > 0 else np.nan))
    return rows

def build_contrasts(fus):
    FULL = ("orig", "P+Z+A", "var_a1"); UNI = ("orig", "P+Z+A", "uniform"); A_ = ("orig", "A", "single")
    L = ("L", "cal_f1max"); out = []
    def add(lbl, a, b, regime_policy=L):
        out.extend(paired_contrast(fus, regime_policy[0], regime_policy[1], a, b, label=lbl))
    add("C1  adaptive (var, a=1) vs uniform averaging  [P+Z+A]", FULL, UNI)
    add("C2  full HFM adaptive vs memory-bank branch alone", FULL, A_)
    add("C3  full HFM uniform vs memory-bank branch alone", UNI, A_)
    add("C4a remove P:  full adaptive vs Z+A adaptive", FULL, ("orig", "Z+A", "var_a1"))
    add("C4b remove Z:  full adaptive vs P+A adaptive", FULL, ("orig", "P+A", "var_a1"))
    add("C4c remove A:  full adaptive vs P+Z adaptive", FULL, ("orig", "P+Z", "var_a1"))
    add("C5a adaptive a=10 vs a=1", ("orig", "P+Z+A", "var_a10"), FULL)
    add("C5b adaptive a=50 vs a=1", ("orig", "P+Z+A", "var_a50"), FULL)
    add("C5c AUROC-weighted (label-aware) vs memory-bank alone", ("orig", "P+Z+A", "auc_w"), A_)
    add("C5d logistic-regression fusion (label-aware) vs memory-bank alone", ("orig", "P+Z+A", "logreg"), A_)
    add("C5e best-branch selection on calibration vs memory-bank alone", ("orig", "P+Z+A", "sel_best"), A_)
    add("C5f logistic-regression fusion vs adaptive a=1", ("orig", "P+Z+A", "logreg"), FULL)
    add("C6a engineered prompt ensemble: P+Zp+A adaptive vs P+Z+A adaptive", ("prompt_ens", "P+Zp+A", "var_a1"), FULL)
    add("C6b engineered prompt ensemble Zp alone vs generic Z alone", ("prompt_ens", "Zp", "single"), ("orig", "Z", "single"))
    add("C7  DINOv2 -> CNN features in the full HFM (CNN minus DINOv2)", ("cnn_swap", "Pc+Z+Aw", "var_a1"), FULL)
    add("C8a memory bank DINOv2-L vs Wide-ResNet-50 PatchCore", A_, ("cnn_swap", "Aw", "single"))
    add("C8b prototype DINOv2-L vs ResNet-50 global", ("orig", "P", "single"), ("cnn_swap", "Pc", "single"))
    add("C9  memory bank vs prototype branch (DINOv2)", A_, ("orig", "P", "single"))
    add("C10 CLIP branch alone vs chance-level 0.5 reference (Z - P)", ("orig", "Z", "single"), ("orig", "P", "single"))
    # normal-only regime
    N = ("N", "q95")
    add("N1  normal-only calibration: uniform z-score vs memory bank alone", ("orig", "P+Z+A", "uniform_z"), ("orig", "A", "single"), N)
    add("N2  normal-only calibration: max-z vs memory bank alone", ("orig", "P+Z+A", "max_z"), ("orig", "A", "single"), N)
    add("N3  normal-only: P+A uniform_z vs memory bank alone", ("orig", "P+A", "uniform_z"), ("orig", "A", "single"), N)
    add("N4  normal-only: engineered prompts, P+Zp+A uniform_z vs P+Z+A uniform_z", ("prompt_ens", "P+Zp+A", "uniform_z"), ("orig", "P+Z+A", "uniform_z"), N)
    for r in out:
        r["metric"] = "AUROC (eval set)"
    df = pd.DataFrame(out)
    if len(df):
        df["wilcoxon_p_holm"] = np.nan
        ok = df["wilcoxon_p"].notna()
        df.loc[ok, "wilcoxon_p_holm"] = holm(df.loc[ok, "wilcoxon_p"].values)
    return df

def variance_confidence_tables(var_df):
    if var_df is None or len(var_df) == 0:
        return pd.DataFrame()
    d = with_all(var_df); rows = []
    for ds, g in d.groupby("dataset"):
        g = g.dropna(subset=["var_cal", "auc_cal", "auc_eval"])
        if len(g) < 5:
            continue
        rho_c, p_c = sps.spearmanr(g["var_cal"], g["auc_cal"]); rho_e, p_e = sps.spearmanr(g["var_cal"], g["auc_eval"])
        agree_c = agree_e = n = 0
        for (cat, seed), h in g.groupby(["category", "seed"]):
            if len(h) < 3: continue
            n += 1; bv = h.loc[h["var_cal"].idxmax(), "branch"]
            agree_c += int(bv == h.loc[h["auc_cal"].idxmax(), "branch"]); agree_e += int(bv == h.loc[h["auc_eval"].idxmax(), "branch"])
        rows.append(dict(dataset=ds, n_points=len(g), spearman_var_vs_auc_cal=rho_c, p_cal=p_c, spearman_var_vs_auc_eval=rho_e, p_eval=p_e,
                         pct_highest_variance_branch_is_best_cal=100 * agree_c / max(n, 1),
                         pct_highest_variance_branch_is_best_eval=100 * agree_e / max(n, 1), chance_pct=100 / 3))
    return pd.DataFrame(rows)

def fusion_weight_table(fus):
    d = fus[(fus.regime == "L") & (fus.family == "orig") & (fus.subset == "P+Z+A") & (fus.rule.isin(["var_a1", "var_a10", "var_a50", "auc_w", "logreg"]))].copy()
    if len(d) == 0:
        return pd.DataFrame()
    w = d["w"].str.split("|", expand=True).astype(float); w.columns = ["w_P", "w_Z", "w_A"]
    d = pd.concat([d[["dataset", "category", "seed", "rule"]].reset_index(drop=True), w.reset_index(drop=True)], axis=1)
    d = with_all(d)
    g = d.groupby(["dataset", "rule"])[["w_P", "w_Z", "w_A"]].agg(["mean", "std"])
    g.columns = ["_".join(c) for c in g.columns]
    return g.reset_index()

DS_LABEL = {"mvtec": "MVTec AD", "visa": "VisA", "btad": "BTAD", "mpdd": "MPDD", "mvtec_ad_2": "MVTec AD 2", "crack": "Concrete cracks", "ALL": "All pooled"}
def dsl(x): return DS_LABEL.get(x, x)

def pm(m, s, d=2):
    if m is None or (isinstance(m, float) and np.isnan(m)): return "n/a"
    if s is None or (isinstance(s, float) and np.isnan(s)): return f"{m:.{d}f}"
    return f"{m:.{d}f} ± {s:.{d}f}"

def pct(df, cols):
    df = df.copy()
    for c in cols:
        if c in df.columns: df[c] = 100 * df[c]
    return df

def sel(summ, regime, policy, family, subset, rule):
    d = summ[(summ.regime == regime) & (summ.policy == policy) & (summ.family == family) & (summ.subset == subset) & (summ.rule == rule)]
    return d.set_index("dataset")

def make_fusion_tables(fus):
    metric_cols = ["auroc_eval", "auroc_full", "acc", "prec", "rec", "f1", "fpr", "oracle_acc", "oracle_prec", "oracle_rec", "oracle_f1", "oracle_fpr"]
    summ = summarize(pct(fus, [c for c in metric_cols if c in fus.columns]), ["regime", "policy", "family", "subset", "rule"],
                     [c for c in metric_cols if c in fus.columns], boot_cols=("auroc_eval", "auroc_full"))
    save_table(summ, "summary_all_configurations.csv")
    return summ

def table_main(summ):
    rows = []
    cfgs = [("Prototype branch P (DINOv2 CLS)", ("orig", "P", "single")), ("CLIP branch Z (generic prompts)", ("orig", "Z", "single")),
            ("CLIP branch Zp (engineered prompt ensemble)", ("prompt_ens", "Zp", "single")),
            ("Memory-bank branch A (DINOv2 patches)", ("orig", "A", "single")),
            ("HFM uniform averaging (P+Z+A)", ("orig", "P+Z+A", "uniform")), ("HFM adaptive variance softmax a=1 (P+Z+A)", ("orig", "P+Z+A", "var_a1")),
            ("HFM + engineered prompts, adaptive (P+Zp+A)", ("prompt_ens", "P+Zp+A", "var_a1")),
            ("Memory + prototype only (P+A), uniform", ("orig", "P+A", "uniform")),
            ("ResNet-50 global baseline", ("cnn_swap", "Pc", "single")), ("PatchCore Wide-ResNet-50", ("cnn_swap", "Aw", "single"))]
    for name, (f, s, r) in cfgs:
        d = summ[(summ.regime == "L") & (summ.policy == "cal_f1max") & (summ.family == f) & (summ.subset == s) & (summ.rule == r)]
        for _, x in d.iterrows():
            rows.append(dict(method=name, dataset=dsl(x.dataset), AUROC=pm(x.auroc_eval, x.auroc_eval_std),
                             AUROC_CI95_over_categories=f"[{x.get('auroc_eval_ci_cat_lo', np.nan):.2f}, {x.get('auroc_eval_ci_cat_hi', np.nan):.2f}]",
                             Accuracy=pm(x.acc, x.acc_std), Precision=pm(x.prec, x.prec_std), Recall=pm(x.rec, x.rec_std),
                             F1=pm(x.f1, x.f1_std), FPR=pm(x.fpr, x.fpr_std),
                             **{"Oracle F1 (test-optimised upper bound)": pm(x.oracle_f1, x.oracle_f1_std)}))
    df = pd.DataFrame(rows); save_table(df, "TABLE_main_results_strict_protocol.csv"); return df

def table_normal_only(summ):
    rows = []
    for (f, s, r, name) in [("orig", "A", "single", "Memory bank A"), ("orig", "P", "single", "Prototype P"), ("orig", "Z", "single", "CLIP Z (generic)"),
                            ("prompt_ens", "Zp", "single", "CLIP Zp (ensemble)"), ("orig", "P+Z+A", "uniform_z", "HFM uniform z-score"),
                            ("orig", "P+Z+A", "max_z", "HFM max z-score"), ("orig", "P+A", "uniform_z", "P+A uniform z-score"),
                            ("prompt_ens", "P+Zp+A", "uniform_z", "HFM+engineered prompts, uniform z"), ("cnn_swap", "Aw", "single", "PatchCore WRN-50"),
                            ("cnn_swap", "Pc", "single", "ResNet-50 global")]:
        for q in THR_QUANTILES:
            d = summ[(summ.regime == "N") & (summ.policy == f"q{q}") & (summ.family == f) & (summ.subset == s) & (summ.rule == r)]
            for _, x in d.iterrows():
                rows.append(dict(method=name, dataset=dsl(x.dataset), threshold=f"{q}th percentile of validation-normal scores",
                                 AUROC_full_test=pm(x.auroc_full, x.auroc_full_std),
                                 AUROC_CI95_over_categories=f"[{x.get('auroc_full_ci_cat_lo', np.nan):.2f}, {x.get('auroc_full_ci_cat_hi', np.nan):.2f}]",
                                 Accuracy=pm(x.acc, x.acc_std), Precision=pm(x.prec, x.prec_std), Recall=pm(x.rec, x.rec_std),
                                 F1=pm(x.f1, x.f1_std), FPR_on_test_normals=pm(x.fpr, x.fpr_std)))
    df = pd.DataFrame(rows); save_table(df, "TABLE_normal_only_calibration_no_defect_labels.csv"); return df

def table_threshold_policies(summ):
    rows = []
    for (f, s, r, name) in [("orig", "P+Z+A", "var_a1", "Full HFM (adaptive)"), ("orig", "A", "single", "Memory bank alone")]:
        for ds in summ.dataset.unique():
            dL = summ[(summ.regime == "L") & (summ.policy == "cal_f1max") & (summ.family == f) & (summ.subset == s) & (summ.rule == r) & (summ.dataset == ds)]
            if len(dL):
                x = dL.iloc[0]
                rows.append(dict(method=name, dataset=dsl(ds), policy="Test-optimised F1-max (upper bound, as in submitted version)", acc=x.oracle_acc, prec=x.oracle_prec, rec=x.oracle_rec, f1=x.oracle_f1, fpr=x.oracle_fpr))
                rows.append(dict(method=name, dataset=dsl(ds), policy="F1-max on separate calibration set (val normals + calibration defects)", acc=x.acc, prec=x.prec, rec=x.rec, f1=x.f1, fpr=x.fpr))
            rn = "uniform_z" if len(s) > 1 else "single"
            for q in THR_QUANTILES:
                dN = summ[(summ.regime == "N") & (summ.policy == f"q{q}") & (summ.family == f) & (summ.subset == s) & (summ.rule == rn) & (summ.dataset == ds)]
                if len(dN):
                    x = dN.iloc[0]
                    rows.append(dict(method=name + (" [normal-only: uniform z-score fusion]" if len(s) > 1 else ""), dataset=dsl(ds), policy=f"Normal-only: {q}th percentile of validation normals", acc=x.acc, prec=x.prec, rec=x.rec, f1=x.f1, fpr=x.fpr))
    df = pd.DataFrame(rows).round(2); save_table(df, "TABLE_threshold_policies.csv"); return df

def table_ablation(summ, contrasts):
    order = [("orig", "A", "single"), ("orig", "P", "single"), ("orig", "Z", "single"), ("orig", "P+Z", "var_a1"), ("orig", "P+A", "var_a1"),
             ("orig", "Z+A", "var_a1"), ("orig", "P+Z+A", "uniform"), ("orig", "P+Z+A", "var_a1"), ("cnn_swap", "Pc", "single"), ("cnn_swap", "Aw", "single"),
             ("cnn_swap", "Pc+Z+Aw", "var_a1"), ("prompt_ens", "P+Zp+A", "var_a1")]
    rows = []
    for ds in summ.dataset.unique():
        ref = summ[(summ.regime == "L") & (summ.policy == "cal_f1max") & (summ.family == "orig") & (summ.subset == "P+Z+A") & (summ.rule == "var_a1") & (summ.dataset == ds)]
        for f, s, r in order:
            d = summ[(summ.regime == "L") & (summ.policy == "cal_f1max") & (summ.family == f) & (summ.subset == s) & (summ.rule == r) & (summ.dataset == ds)]
            if len(d) == 0 or len(ref) == 0: continue
            x = d.iloc[0]
            rows.append(dict(dataset=dsl(ds), configuration=f"{f}: {s} [{r}]", AUROC=pm(x.auroc_eval, x.auroc_eval_std), F1=pm(x.f1, x.f1_std),
                             delta_AUROC_vs_full_HFM_pp=round(float(x.auroc_eval - ref.iloc[0].auroc_eval), 2)))
    df = pd.DataFrame(rows); save_table(df, "TABLE_ablation_leave_one_out_and_single_branches.csv"); return df

def table_fusion_rules(summ):
    rows = []
    for ds in summ.dataset.unique():
        for rule in L_RULES:
            d = summ[(summ.regime == "L") & (summ.policy == "cal_f1max") & (summ.family == "orig") & (summ.subset == "P+Z+A") & (summ.rule == rule) & (summ.dataset == ds)]
            if len(d):
                x = d.iloc[0]; rows.append(dict(dataset=dsl(ds), rule=rule, AUROC=pm(x.auroc_eval, x.auroc_eval_std), F1=pm(x.f1, x.f1_std)))
        d = summ[(summ.regime == "L") & (summ.policy == "cal_f1max") & (summ.family == "orig") & (summ.subset == "A") & (summ.dataset == ds)]
        if len(d):
            x = d.iloc[0]; rows.append(dict(dataset=dsl(ds), rule="(memory bank alone)", AUROC=pm(x.auroc_eval, x.auroc_eval_std), F1=pm(x.f1, x.f1_std)))
    df = pd.DataFrame(rows); save_table(df, "TABLE_fusion_rules_P+Z+A.csv"); return df

def fig_confusion(fus):
    d0 = fus[(fus.seed == 0) & (fus.family == "orig") & (fus.subset == "P+Z+A")]
    pairs = [("Calibration regime (F1-max on calibration set)", d0[(d0.regime == "L") & (d0.rule == "var_a1")]),
             ("Normal-only regime (95th percentile of validation normals)", d0[(d0.regime == "N") & (d0.policy == "q95") & (d0.rule == "uniform_z")])]
    dss = [x for x in fus.dataset.unique()]
    fig, axes = plt.subplots(2, len(dss), figsize=(3.3 * len(dss), 6.6), squeeze=False)
    for r, (ttl, dd) in enumerate(pairs):
        for c, ds in enumerate(dss):
            g = dd[dd.dataset == ds]; ax = axes[r, c]
            cm = np.array([[g.tn.sum(), g.fp.sum()], [g.fn.sum(), g.tp.sum()]])
            ax.imshow(cm, cmap="Blues")
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=11)
            ax.set_xticks([0, 1]); ax.set_xticklabels(["Normal", "Defect"]); ax.set_yticks([0, 1]); ax.set_yticklabels(["Normal", "Defect"])
            ax.set_title(dsl(ds) if r == 0 else "", fontweight="bold"); ax.set_xlabel("Predicted"); ax.set_ylabel("True" if c == 0 else "")
            if c == 0: ax.annotate(ttl, xy=(0, 1.18), xycoords="axes fraction", fontsize=8, color=PAL["grey"])
    fig.tight_layout(); save_fig(fig, "fig_confusion_matrices_strict_protocol.png")

def fig_roc(curves):
    if not curves: return
    dss = sorted(set(c["dataset"] for c in curves)); keys = [("A|single", "Memory bank alone", PAL["blue"]), ("P+Z+A|uniform", "HFM uniform", PAL["orange"]), ("P+Z+A|var_a1", "HFM adaptive", PAL["verm"])]
    key_map = {"A|single": "A|single"}
    fig, axes = plt.subplots(1, len(dss), figsize=(3.4 * len(dss), 3.5), squeeze=False); grid = np.linspace(0, 1, 201)
    for c, ds in enumerate(dss):
        ax = axes[0, c]
        for k, lbl, col in keys:
            tprs = []
            for cv in curves:
                if cv["dataset"] == ds and cv["key"] == k and len(np.unique(cv["y"])) == 2:
                    fpr, tpr, _ = roc_curve(cv["y"], cv["s"]); tprs.append(np.interp(grid, fpr, tpr))
            if tprs:
                m = np.mean(tprs, 0); ax.plot(grid, m, color=col, lw=1.8, label=f"{lbl} (AUC {100 * np.trapezoid(m, grid) if hasattr(np, 'trapezoid') else 100 * np.trapz(m, grid):.1f})")
        ax.plot([0, 1], [0, 1], "--", color=PAL["grey"], lw=0.8); ax.set_title(dsl(ds), fontweight="bold")
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate" if c == 0 else ""); ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout(); save_fig(fig, "fig_roc_macro_average_strict_protocol.png")

def fig_ablation(summ):
    cfgs = [("A", ("orig", "A", "single")), ("P", ("orig", "P", "single")), ("Z", ("orig", "Z", "single")), ("P+Z", ("orig", "P+Z", "var_a1")),
            ("P+A", ("orig", "P+A", "var_a1")), ("Z+A", ("orig", "Z+A", "var_a1")), ("P+Z+A\nuniform", ("orig", "P+Z+A", "uniform")), ("P+Z+A\nadaptive", ("orig", "P+Z+A", "var_a1"))]
    cols = [PAL["blue"]] * 3 + [PAL["orange"]] * 3 + [PAL["green"], PAL["verm"]]
    dss = [d for d in summ.dataset.unique() if d != "ALL"] + ["ALL"]
    fig, axes = plt.subplots(1, len(dss), figsize=(3.6 * len(dss), 3.6), squeeze=False, sharey=True)
    for c, ds in enumerate(dss):
        ax = axes[0, c]; vals, errs = [], []
        for _, (f, s, r) in cfgs:
            d = summ[(summ.regime == "L") & (summ.policy == "cal_f1max") & (summ.family == f) & (summ.subset == s) & (summ.rule == r) & (summ.dataset == ds)]
            vals.append(d.auroc_eval.iloc[0] if len(d) else np.nan); errs.append(d.auroc_eval_std.iloc[0] if len(d) else 0)
        ax.bar(range(len(cfgs)), vals, yerr=np.nan_to_num(errs), color=cols, capsize=2, width=0.75)
        ax.set_xticks(range(len(cfgs))); ax.set_xticklabels([n for n, _ in cfgs], rotation=60, ha="right", fontsize=7)
        ax.set_title(dsl(ds), fontweight="bold"); ax.set_ylim(40, 100); ax.grid(axis="y", ls=":", alpha=.5)
        if c == 0: ax.set_ylabel("Image AUROC (%)  (mean ± std over seeds)")
    fig.tight_layout(); save_fig(fig, "fig_ablation_single_branches_and_leave_one_out.png")

def fig_fusion_rules(summ):
    dss = [d for d in summ.dataset.unique() if d != "ALL"] + ["ALL"]
    fig, ax = plt.subplots(figsize=(1.9 * len(dss) + 2, 3.6)); w = 0.8 / (len(L_RULES) + 1)
    cols = [PAL["blue"], PAL["sky"], PAL["green"], PAL["yellow"], PAL["orange"], PAL["verm"], PAL["pink"], PAL["grey"], PAL["black"]]
    for i, rule in enumerate(L_RULES + ["A"]):
        v = []
        for ds in dss:
            if rule == "A": d = summ[(summ.regime == "L") & (summ.family == "orig") & (summ.subset == "A") & (summ.dataset == ds)]
            else: d = summ[(summ.regime == "L") & (summ.family == "orig") & (summ.subset == "P+Z+A") & (summ.rule == rule) & (summ.dataset == ds)]
            v.append(d.auroc_eval.iloc[0] if len(d) else np.nan)
        ax.bar(np.arange(len(dss)) + i * w, v, w, label=("memory bank alone" if rule == "A" else rule), color=cols[i % len(cols)])
    ax.set_xticks(np.arange(len(dss)) + 0.4 - w / 2); ax.set_xticklabels([dsl(d) for d in dss]); ax.set_ylim(40, 100); ax.set_ylabel("Image AUROC (%)")
    ax.legend(fontsize=7, ncol=3, loc="lower center"); ax.grid(axis="y", ls=":", alpha=.5); fig.tight_layout(); save_fig(fig, "fig_fusion_rules_comparison.png")

def fig_thresholds(tp):
    if len(tp) == 0: return
    d = tp[tp.method.str.startswith("Full HFM")]; dss = d.dataset.unique(); pols = d.policy.unique()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6)); cols = [PAL["grey"], PAL["blue"], PAL["orange"], PAL["verm"]]
    for ax, (col, ttl) in zip(axes, [("rec", "Recall (defect detection rate), %"), ("fpr", "False-positive rate on normal images, %")]):
        w = 0.8 / len(pols)
        for i, p in enumerate(pols):
            v = [d[(d.dataset == ds) & (d.policy == p)][col].mean() for ds in dss]
            ax.bar(np.arange(len(dss)) + i * w, v, w, color=cols[i % 4], label=p.replace("Normal-only: ", "Normal-only ")[:48])
        ax.set_xticks(np.arange(len(dss)) + 0.4 - w / 2); ax.set_xticklabels(dss, rotation=20); ax.set_title(ttl); ax.grid(axis="y", ls=":", alpha=.5)
    axes[0].legend(fontsize=6, loc="upper center", bbox_to_anchor=(1.1, -0.25), ncol=2); fig.tight_layout(); save_fig(fig, "fig_threshold_policies.png")

def fig_fewshot(fs, ns):
    if len(fs) == 0 and len(ns) == 0: return
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6)); cols = list(PAL.values())
    if len(fs):
        s = summarize(fs, ["k"], ["acc", "auroc"], pooled=False)
        for i, ds in enumerate(s.dataset.unique()):
            g = s[s.dataset == ds].sort_values("k")
            for ax, m in zip(axes[:2], ["acc", "auroc"]):
                ax.plot(g.k, g[m], marker="o", color=cols[i % 6], label=dsl(ds)); ax.fill_between(g.k, g[m] - g[m + "_std"], g[m] + g[m + "_std"], color=cols[i % 6], alpha=.15)
        axes[0].set_title("Prototype classifier, k normal + k defective\n(natural nearest-prototype rule), accuracy %"); axes[1].set_title("Prototype classifier, AUROC %")
    if len(ns):
        s = summarize(ns, ["k"], ["auroc_bank_top3"], pooled=False)
        for i, ds in enumerate(s.dataset.unique()):
            g = s[s.dataset == ds].sort_values("k"); axes[2].plot(g.k, g.auroc_bank_top3, marker="s", color=cols[i % 6], label=dsl(ds))
            axes[2].fill_between(g.k, g.auroc_bank_top3 - g.auroc_bank_top3_std, g.auroc_bank_top3 + g.auroc_bank_top3_std, color=cols[i % 6], alpha=.15)
        axes[2].set_title("Normal-only k-shot memory bank\n(AnomalyDINO-style), AUROC %")
    for ax in axes: ax.set_xlabel("shots k"); ax.set_xscale("log"); ax.grid(ls=":", alpha=.5)
    axes[0].legend(fontsize=7); fig.tight_layout(); save_fig(fig, "fig_fewshot_curves.png")

def fig_zeroshot(zs):
    if len(zs) == 0: return
    g = zs.groupby("dataset")[["auroc_generic", "auroc_ensemble", "auroc_oracle_names", "top1", "chance", "defect_only_acc", "defect_only_chance"]].mean()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6)); w = 0.27; x = np.arange(len(g))
    for i, (c, l, col) in enumerate([("auroc_generic", "generic prompts (fused model)", PAL["blue"]), ("auroc_ensemble", "engineered prompt ensemble", PAL["green"]), ("auroc_oracle_names", "prompts with test-annotation defect names", PAL["orange"])]):
        axes[0].bar(x + i * w, g[c], w, label=l, color=col)
    axes[0].axhline(50, color=PAL["grey"], ls="--", lw=1); axes[0].set_xticks(x + w); axes[0].set_xticklabels([dsl(d) for d in g.index], rotation=20); axes[0].set_ylabel("Binary detection AUROC (%)"); axes[0].legend(fontsize=7)
    axes[1].bar(x - 0.2, g["top1"], 0.4, label="top-1, all classes incl. normal", color=PAL["blue"]); axes[1].bar(x + 0.2, g["chance"], 0.4, label="chance level", color=PAL["grey"])
    axes[1].set_xticks(x); axes[1].set_xticklabels([dsl(d) for d in g.index], rotation=20); axes[1].set_ylabel("Defect-name recognition accuracy (%)"); axes[1].legend(fontsize=7)
    fig.tight_layout(); save_fig(fig, "fig_zero_shot.png")

def fig_variance(var_df):
    if len(var_df) == 0: return
    fig, ax = plt.subplots(figsize=(4.8, 4)); col = {"P": PAL["blue"], "Z": PAL["orange"], "A": PAL["green"]}
    for b in ("P", "Z", "A"):
        g = var_df[var_df.branch == b]; ax.scatter(g.var_cal, 100 * g.auc_eval, s=8, alpha=.35, color=col[b], label=f"branch {b}")
    ax.axhline(50, color=PAL["grey"], ls="--", lw=.8); ax.set_xlabel("Variance of min–max normalised scores (calibration set)"); ax.set_ylabel("Branch AUROC on evaluation set (%)")
    ax.legend(fontsize=8); ax.grid(ls=":", alpha=.5); fig.tight_layout(); save_fig(fig, "fig_variance_vs_branch_auroc.png")

def fig_bank(bank):
    if len(bank) == 0: return
    s = summarize(bank, ["ratio"], ["auroc", "ms_per_image"], pooled=False); fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4)); cols = list(PAL.values())
    for i, ds in enumerate(s.dataset.unique()):
        g = s[s.dataset == ds].sort_values("ratio")
        axes[0].plot(100 * g.ratio, g.auroc, marker="o", color=cols[i % 6], label=dsl(ds)); axes[1].plot(100 * g.ratio, g.ms_per_image, marker="o", color=cols[i % 6])
    axes[0].set_xlabel("Memory-bank size (% of the 40,000-patch pool)"); axes[0].set_ylabel("Image AUROC (%)"); axes[1].set_xlabel("Memory-bank size (% of the pool)"); axes[1].set_ylabel("NN-search time (ms / image)")
    axes[0].legend(fontsize=7); [a.grid(ls=":", alpha=.5) for a in axes]; fig.tight_layout(); save_fig(fig, "fig_memory_bank_size.png")

def fig_robust(rob):
    if len(rob) == 0: return
    s = summarize(rob, ["corruption", "method"], ["auroc"], pooled=False)
    dss = s.dataset.unique(); fig, axes = plt.subplots(1, len(dss), figsize=(5.6 * len(dss), 3.8), squeeze=False)
    meths = ["A", "P", "Z", "HFM_uniform", "HFM_adaptive"]; cols = [PAL["blue"], PAL["green"], PAL["orange"], PAL["sky"], PAL["verm"]]; w = 0.8 / len(meths)
    corrs = ["clean"] + ROB_CORRUPTIONS
    for c, ds in enumerate(dss):
        ax = axes[0, c]
        for i, m in enumerate(meths):
            v = [s[(s.dataset == ds) & (s.method == m) & (s.corruption == k)].auroc.mean() for k in corrs]
            ax.bar(np.arange(len(corrs)) + i * w, v, w, color=cols[i], label=m)
        ax.set_xticks(np.arange(len(corrs)) + 0.4 - w / 2); ax.set_xticklabels(corrs, rotation=35, ha="right", fontsize=8); ax.set_ylim(max(0, min(30, np.nanmin(s.auroc) - 5)), 100)
        ax.set_title(dsl(ds), fontweight="bold"); ax.set_ylabel("Image AUROC (%)" if c == 0 else ""); ax.grid(axis="y", ls=":", alpha=.5)
    axes[0, 0].legend(fontsize=7, ncol=2); fig.tight_layout(); save_fig(fig, "fig_robustness_to_inspection_conditions.png")

def fig_localization(loc):
    if len(loc) == 0: return
    s = summarize(loc[loc.smoothing == "gauss4"], [], ["pixel_auroc", "aupro03", "pixel_ap", "dice", "region_f1_iou0.1", "boundary_f"], pooled=False)
    mets = [("pixel_auroc", "Pixel AUROC"), ("aupro03", "AU-PRO@0.3"), ("pixel_ap", "Pixel AP"), ("dice", "Dice"), ("region_f1_iou0.1", "Region F1 (IoU≥0.1)"), ("boundary_f", "Boundary F (3 px)")]
    fig, ax = plt.subplots(figsize=(1.7 * len(mets) + 2, 3.6)); cols = list(PAL.values()); dss = s.dataset.unique(); w = 0.8 / len(dss)
    for i, ds in enumerate(dss):
        g = s[s.dataset == ds]; ax.bar(np.arange(len(mets)) + i * w, [g[m].iloc[0] for m, _ in mets], w, yerr=[np.nan_to_num(g[m + "_std"].iloc[0]) for m, _ in mets], color=cols[i % 6], label=dsl(ds), capsize=2)
    ax.set_xticks(np.arange(len(mets)) + 0.4 - w / 2); ax.set_xticklabels([n for _, n in mets], rotation=20); ax.set_ylabel("%"); ax.legend(fontsize=7); ax.grid(axis="y", ls=":", alpha=.5)
    fig.tight_layout(); save_fig(fig, "fig_localization_metrics.png")

def fig_loc_qualitative(qual):
    by = defaultdict(list)
    for (ds, cat), v in qual.items():
        if v["ex"]: by[ds].append((cat, v))
    for ds, lst in by.items():
        lst = sorted(lst, key=lambda t: t[1]["auc"]); pick = [lst[0], lst[len(lst) // 2], lst[-1]] if len(lst) >= 3 else lst
        cols = [(cat, e) for cat, v in pick for e in v["ex"]]
        if not cols: continue
        fig, axes = plt.subplots(5, len(cols), figsize=(2.0 * len(cols), 10.4), squeeze=False)
        for j, (cat, e) in enumerate(cols):
            img = center_crop(resize_shorter(Image.open(e["path"]).convert("RGB"), RESIZE_SIZE, Image.BILINEAR), IMG_SIZE); im = np.asarray(img)
            gt = load_gt_mask(e["mask"], "aligned").astype(bool) if e["mask"] and os.path.exists(e["mask"]) else np.zeros((IMG_SIZE, IMG_SIZE), bool)
            hm = upsample_maps(e["map"][None], SMOOTH_SIGMA)[0].cpu().numpy(); tau = e["tau"]; pred = hm > tau
            err = np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8); err[pred & gt] = (0, 158, 115); err[pred & ~gt] = (213, 94, 0); err[~pred & gt] = (0, 114, 178)
            ov = im.copy(); ov[gt] = (0.5 * ov[gt] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
            panels = [im, ov, hm, pred, err]
            for i, p in enumerate(panels):
                ax = axes[i, j]; ax.axis("off")
                if i == 2: ax.imshow(im); ax.imshow(p, cmap="jet", alpha=0.5, vmin=min(hm.min(), tau * 0.5), vmax=max(hm.max(), tau * 1.2))
                elif i == 3: ax.imshow(p, cmap="gray", vmin=0, vmax=1)
                else: ax.imshow(p)
            axes[0, j].set_title(f"{cat}\n{e['kind'].replace('Highest-scoring ','top ').replace('Lowest-scoring ','lowest ')} ({e['score']:.2f})", fontsize=7)
        for i, n in enumerate(["Image", "Ground truth (red)", "Anomaly map", "Binary prediction", "TP green / FP orange / FN blue"]):
            axes[i, 0].text(-0.05, 0.5, n, rotation=90, va="center", ha="right", transform=axes[i, 0].transAxes, fontsize=7)
        fig.suptitle(f"{dsl(ds)}: localisation vs. ground truth (aligned masks), worst / median / best category by image AUROC", fontsize=9)
        fig.tight_layout(); save_fig(fig, f"fig_localization_qualitative_{ds}.png", dpi=150)

def fig_efficiency(eff, scaling_tbl):
    if not eff: return
    lat = eff["latency"]; fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    d = lat.iloc[:6]; axes[0].barh(d.stage[::-1], d.mean_ms[::-1], xerr=d.std_ms[::-1], color=PAL["blue"]); axes[0].set_xlabel("ms per image (batch size 1, mean ± std)")
    axes[0].set_title("Latency by stage (T4)"); axes[0].grid(axis="x", ls=":", alpha=.5)
    if scaling_tbl is not None and len(scaling_tbl):
        for _, r in scaling_tbl.iterrows():
            if pd.notna(r.get("latency_ms", np.nan)):
                axes[1].scatter(r.latency_ms, r.A_auroc, s=60, color=PAL["verm"] if "L/14" in r.backbone else PAL["blue"]); axes[1].annotate(r.backbone, (r.latency_ms, r.A_auroc), fontsize=7, xytext=(4, 4), textcoords="offset points")
        axes[1].set_xlabel("Backbone forward latency (ms, batch 1)"); axes[1].set_ylabel("Image AUROC on MVTec AD (%)"); axes[1].set_title("Accuracy-latency trade-off"); axes[1].grid(ls=":", alpha=.5)
    fig.tight_layout(); save_fig(fig, "fig_efficiency.png")

# MAIN
def result_path(ds, cat):
    safe = f"{ds}__{cat}".replace("/", "_").replace(" ", "_")
    return os.path.join(CACHE_DIR, "results", f"script{SCRIPT_ID}__{'quick__' if QUICK_TEST else ''}{safe}.pkl")

def df_from(all_res, key):
    rows = [r for res in all_res for r in res.get(key, []) if isinstance(r, dict)]
    return pd.DataFrame(rows)

def md(df, n=None, digits=2):
    if df is None or len(df) == 0: return "(empty)"
    d = df.head(n) if n else df
    try: return d.round(digits).to_markdown(index=False)
    except Exception: return d.to_string(index=False)

def finalize(all_res, eff, tasks):
    log("\n" + "=" * 80 + "\nAGGREGATION, STATISTICS, TABLES AND FIGURES\n" + "=" * 80)
    fus = df_from(all_res, "fusion"); var = df_from(all_res, "variance"); fs = df_from(all_res, "fewshot"); ns = df_from(all_res, "normalshot")
    zs = df_from(all_res, "zeroshot"); bank = df_from(all_res, "bank"); loc = df_from(all_res, "loc"); rob = df_from(all_res, "robust")
    scl = df_from(all_res, "scaling"); spl = df_from(all_res, "split"); tim = df_from(all_res, "timing")
    curves = [c for res in all_res for c in res.get("curves", [])]
    for nm, d in [("fusion_long", fus), ("variance_long", var), ("fewshot_long", fs), ("normalshot_long", ns), ("zeroshot_per_category", zs),
                  ("bank_size_long", bank), ("localization_long", loc), ("robustness_long", rob), ("backbone_scaling_long", scl)]:
        if len(d): d.to_csv(os.path.join(OUT_DIR, "raw", nm + ".csv.gz"), index=False, compression="gzip")
    if len(fus) == 0:
        log("[ERROR] no fusion results were produced."); return {}
    out = {}
    save_table(spl, "TABLE_protocol_split_sizes_seed0.csv")
    save_table(tim, "timing_per_category.csv")
    summ = make_fusion_tables(fus)
    out["main"] = table_main(summ); out["normal_only"] = table_normal_only(summ); out["thr"] = table_threshold_policies(summ)
    out["ablation"] = table_ablation(summ, None); out["rules"] = table_fusion_rules(summ)
    log("computing paired statistics ...")
    con = build_contrasts(fus); save_table(con, "TABLE_paired_statistics_contrasts.csv"); out["con"] = con
    vt = variance_confidence_tables(var); save_table(vt, "TABLE_variance_as_confidence_check.csv"); out["var"] = vt
    wt = fusion_weight_table(fus); save_table(wt, "TABLE_fusion_weights.csv"); out["w"] = wt
    # few-shot
    if len(fs):
        a = summarize(fs, ["k"], ["auroc", "acc", "f1", "prec", "rec", "oracle_acc", "oracle_f1"], boot_cols=("auroc",), pooled=True); save_table(a, "TABLE_fewshot_prototype_normal_plus_defect.csv"); out["fs"] = a
    if len(ns):
        b = summarize(ns, ["k"], ["auroc_proto_normal", "auroc_bank_top3", "auroc_bank_top1pct"], boot_cols=("auroc_bank_top3",), pooled=True); save_table(b, "TABLE_fewshot_normal_only_memory_bank.csv"); out["ns"] = b
    if len(zs):
        z = zs.groupby("dataset")[["n_classes", "top1", "top5", "chance", "majority_class", "defect_only_acc", "defect_only_chance", "auroc_generic", "auroc_ensemble", "auroc_oracle_names"]].mean().reset_index()
        save_table(z, "TABLE_zero_shot_summary.csv"); out["zs"] = z; save_table(zs, "TABLE_zero_shot_per_category.csv")
    if len(loc):
        mcols = ["pixel_auroc", "pixel_auroc_legacy_mask", "pixel_ap", "aupro03", "dice", "iou", "pix_prec", "pix_rec", "boundary_f",
                 "region_prec_iou0.1", "region_rec_iou0.1", "region_f1_iou0.1", "region_prec_iou0.3", "region_rec_iou0.3", "region_f1_iou0.3"]
        l = summarize(loc, ["smoothing"], mcols, pooled=True); save_table(l, "TABLE_localization_metrics.csv"); out["loc"] = l
        save_table(loc, "TABLE_localization_per_category.csv")
        fix = l[l.smoothing == "raw"][["dataset", "pixel_auroc_legacy_mask", "pixel_auroc"]].rename(columns={"pixel_auroc_legacy_mask": "pixel_AUROC_misaligned_masks_(submitted_protocol)", "pixel_auroc": "pixel_AUROC_aligned_masks_(corrected)"})
        fix["difference_pp"] = fix.iloc[:, 2] - fix.iloc[:, 1]; save_table(fix, "TABLE_localization_mask_alignment_effect.csv"); out["fix"] = fix
    if len(rob):
        r = summarize(rob, ["corruption", "method"], ["auroc"], pooled=False)
        clean = r[r.corruption == "clean"].set_index(["dataset", "method"])["auroc"]
        r["drop_vs_clean_pp"] = [x.auroc - clean.get((x.dataset, x.method), np.nan) for x in r.itertuples()]
        save_table(r, "TABLE_robustness_inspection_conditions.csv"); out["rob"] = r
    if len(bank):
        bk = summarize(bank, ["ratio"], ["auroc", "ms_per_image", "bank_size"], pooled=False); save_table(bk, "TABLE_memory_bank_size.csv"); out["bank"] = bk
    scaling_tbl = None
    if len(scl):
        s = summarize(scl, ["backbone"], ["A_auroc", "P_auroc"], pooled=False)
        for ds in s.dataset.unique():
            for sub, name, col in [("A", "dinov2_vitl14", "A"), ("P", "dinov2_vitl14", "P"), ("Aw", "wrn50", "A"), ("Pc", "resnet50", "P")]:
                pass
            row = {}
            for bname, sub in [("dinov2_vitl14", ("orig", "A")), ("wrn50_patchcore", ("cnn_swap", "Aw"))]:
                d = summ[(summ.regime == "N") & (summ.policy == "q95") & (summ.family == sub[0]) & (summ.subset == sub[1]) & (summ.rule == "single") & (summ.dataset == ds)]
                if len(d): row[bname] = d.auroc_full.iloc[0]
            dp = summ[(summ.regime == "N") & (summ.policy == "q95") & (summ.family == "orig") & (summ.subset == "P") & (summ.rule == "single") & (summ.dataset == ds)]
            for bname, v in row.items():
                s = pd.concat([s, pd.DataFrame([dict(dataset=ds, backbone=bname, A_auroc=v, P_auroc=dp.auroc_full.iloc[0] if (bname == "dinov2_vitl14" and len(dp)) else np.nan)])], ignore_index=True)
        if eff:
            lm = dict(zip(eff["backbones_latency"].model, eff["backbones_latency"].mean_ms))
            m = {"dinov2_vits14": "DINOv2 dinov2_vits14 @336 (bank branch)", "dinov2_vitb14": "DINOv2 dinov2_vitb14 @336 (bank branch)",
                 "dinov2_vitl14": "DINOv2-L/14 @336 (bank branch)", "wrn50_patchcore": "Wide-ResNet-50-2 PatchCore @224"}
            s["latency_ms"] = s.backbone.map(lambda b: lm.get(m.get(b, ""), np.nan))
        s["backbone"] = s.backbone.replace({"dinov2_vits14": "DINOv2 ViT-S/14", "dinov2_vitb14": "DINOv2 ViT-B/14", "dinov2_vitl14": "DINOv2 ViT-L/14", "wrn50_patchcore": "Wide-ResNet-50-2 PatchCore"})
        save_table(s, "TABLE_backbone_scaling_accuracy_latency.csv"); out["scl"] = s; scaling_tbl = s[s.dataset == s.dataset.iloc[0]]
    for f, a in [(fig_confusion, (fus,)), (fig_roc, (curves,)), (fig_ablation, (summ,)), (fig_fusion_rules, (summ,)), (fig_thresholds, (out["thr"],)),
                 (fig_fewshot, (fs, ns)), (fig_zeroshot, (zs,)), (fig_variance, (var,)), (fig_bank, (bank,)), (fig_robust, (rob,)),
                 (fig_localization, (loc,)), (fig_efficiency, (eff, scaling_tbl))]:
        try: f(*a)
        except Exception as e: log(f"[warn] figure {f.__name__} failed: {e}"); log(traceback.format_exc())
    try:
        qual = {}
        for res in all_res:
            if res.get("qual"):
                ds, cat = res["split"][0]["dataset"], res["split"][0]["category"]
                auc = fus[(fus.dataset == ds) & (fus.category == cat) & (fus.seed == 0) & (fus.regime == "L") & (fus.subset == "A")].auroc_eval
                tau = res["qual_meta"][0]["tau"]
                qual[(ds, cat)] = dict(ex=[dict(e, tau=tau) for e in res["qual"]], tau=tau, auc=float(auc.iloc[0]) if len(auc) else 0.0)
        fig_loc_qualitative(qual)
    except Exception as e:
        log(f"[warn] qualitative figure failed: {e}"); log(traceback.format_exc())
    return out

def write_summary_md(out, eff):
    L_ = ["# Results summary (auto-generated)", f"Script {SCRIPT_ID}; datasets: {', '.join(DATASETS_TO_RUN)}; seeds: {N_SEEDS}; quick-test: {QUICK_TEST}", ""]
    def sec(title, df, n=None):
        L_.extend([f"## {title}", md(df, n), ""])
    sec("Main results, strict protocol (calibration regime)", out.get("main"))
    sec("Normal-only calibration (no defect label used anywhere)", out.get("normal_only"))
    sec("Threshold policies", out.get("thr"))
    sec("Ablation (single branches, leave-one-out, CNN swap)", out.get("ablation"))
    sec("Fusion rules (P+Z+A)", out.get("rules"))
    sec("Paired statistics", out.get("con"))
    sec("Variance as a confidence measure", out.get("var"))
    sec("Fusion weights", out.get("w"))
    sec("Few-shot prototype (normal + defective supports)", out.get("fs")); sec("Few-shot normal-only", out.get("ns"))
    sec("Zero-shot", out.get("zs")); sec("Localisation", out.get("loc")); sec("Mask alignment effect", out.get("fix"))
    sec("Robustness", out.get("rob")); sec("Memory bank size", out.get("bank")); sec("Backbone scaling", out.get("scl"))
    if eff:
        sec("Efficiency: parameters / FLOPs", eff["components"]); sec("Efficiency summary", eff["summary"]); sec("Latency by stage", eff["latency"])
        sec("Backbone latency", eff["backbones_latency"]); sec("Throughput", eff["throughput"])
    open(os.path.join(OUT_DIR, "RESULTS_SUMMARY.md"), "w").write("\n".join(L_))

def make_zip():
    try:
        os.system(f"{sys.executable} -m pip freeze > {OUT_DIR}/requirements_frozen.txt")
    except Exception: pass
    json.dump(env_report(), open(os.path.join(OUT_DIR, "environment.json"), "w"), indent=2, default=str)
    try:
        os.makedirs(os.path.join(OUT_DIR, "code"), exist_ok=True); shutil.copy(__file__, os.path.join(OUT_DIR, "code"))
    except Exception: pass
    base = ZIP_PATH[:-4]
    if os.path.exists(ZIP_PATH): os.remove(ZIP_PATH)
    shutil.make_archive(base, "zip", OUT_DIR)
    log(f"\nZIP READY: {ZIP_PATH}  ({os.path.getsize(ZIP_PATH) / 2**20:.1f} MB)")
    if USE_DRIVE_BACKUP:
        try:
            from google.colab import drive; drive.mount("/content/drive"); shutil.copy(ZIP_PATH, "/content/drive/MyDrive/")
            log("copied to Google Drive")
        except Exception as e: log("[warn] drive backup failed:", e)
    try:
        from google.colab import files; files.download(ZIP_PATH)
    except Exception:
        log("(automatic download not available: download the zip from the Colab file panel)")

def main():
    global CACHE_DIR
    t0 = time.perf_counter(); set_global_seed(0)
    if USE_DRIVE_BACKUP:
        try:
            from google.colab import drive; drive.mount("/content/drive")
            CACHE_DIR = "/content/drive/MyDrive/hfm_cache"; os.makedirs(os.path.join(CACHE_DIR, "results"), exist_ok=True)
            log("checkpoints are stored on Google Drive:", CACHE_DIR)
        except Exception as e:
            log("[warn] Google Drive could not be mounted, using local checkpoints:", e)
    log(json.dumps(env_report(), indent=1, default=str))
    if DEVICE.type != "cuda": log("[warn] no GPU found: switch Runtime > Change runtime type > T4 GPU (CPU will be extremely slow)")
    tasks = collect_all_dataset_tasks()
    if not tasks:
        log("[ERROR] no categories found, nothing to do."); return
    nets = load_nets()
    scale_nets = {}
    if any(d in BACKBONE_SCALING_DATASETS for d in DATASETS_TO_RUN):
        log("Loading DINOv2 ViT-S/14 and ViT-B/14 for the backbone-scaling study...")
        scale_nets = {"dinov2_vits14": DinoNet("dinov2_vits14"), "dinov2_vitb14": DinoNet("dinov2_vitb14")}
    eff = None
    if RUN_EFFICIENCY:
        try:
            eff = run_efficiency(nets, scale_nets, tasks[0][3][0]); cleanup()
        except Exception as e:
            log(f"[warn] efficiency benchmark failed: {e}"); log(traceback.format_exc())
    all_res = []; errors = []
    for i, task in enumerate(tasks):
        ds, cat = task[0], task[1]; rp = result_path(ds, cat)
        log(f"\n[{i + 1}/{len(tasks)}] {ds}/{cat}   (elapsed {(time.perf_counter() - t0) / 60:.1f} min)")
        if os.path.exists(rp):
            try:
                all_res.append(pickle.load(open(rp, "rb"))); log("  loaded from checkpoint"); continue
            except Exception: pass
        try:
            res = process_category(task, nets, scale_nets)
            pickle.dump(res, open(rp, "wb")); all_res.append(res)
            log(f"  done in {res['timing'][0]['seconds']:.0f}s")
        except Exception as e:
            log(f"[ERROR] {ds}/{cat} failed: {e}"); log(traceback.format_exc()); errors.append(f"{ds}/{cat}: {e}"); cleanup()
    if errors:
        open(os.path.join(OUT_DIR, "errors.txt"), "w").write("\n".join(errors))
    out = finalize(all_res, eff, tasks)
    try: write_summary_md(out, eff)
    except Exception as e: log("[warn] summary failed:", e)
    log(f"\nTotal time: {(time.perf_counter() - t0) / 60:.1f} min")
    make_zip()

if __name__ == "__main__":
    main()
