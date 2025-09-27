#  Copyright 2025 Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

# This code uses the surface-distance library by DeepMind.
# Installation instructions available at: https://github.com/deepmind/surface-distance
"""
Evaluate 3D segmentation performance for all subjects in pred_dir,
using only .mha and .nii.gz files. This script:
 - Loads 3D masks and extracts voxel spacing.
 - Ensures prediction masks are binary (0 and 1).
 - If a prediction mask is uniform (all zeros or all ones), all metrics are set to the lowest value possible.
 - Computes surface-based metrics (Dice, Surface Dice at 5mm, Robust Hausdorff95, MASD).
 - Computes tumor volumes and later aggregates metrics (mean for most and RMSE for volumes).
"""

import os
import json
import numpy as np
from pathlib import Path
import SimpleITK as sitk
from surface_distance import metrics as surface_metrics


ALLOWED_EXTENSIONS = [".mha", ".nii.gz"]
panther_msg = """\n
<Computing PANTHER Evaluation Metrics>

                  /)-._
                 Y. ' _]
          ,.._   |`--"=
         /    "-/   \\
/)      |   |_     `\|___
\:::::::\___/_\__\_______\\
"""
panther_msg2 = """\n
  _____________________________
  < PANTHER Evaluation Done! >
  -----------------------------
                                                           _...---.._
                                                       _.'`       -_ ``.
                                                   .-'`                 `.
                                                .-`                     q ;
                                             _-`                       __  \\
                                         .-'`                  . ' .   \ `;/
                                     _.-`                    /.      `._`/
                             _...--'`                        \_`..._
                          .'`                         -         `'--:._
                       .-`                           \                  `-.
                      .                `              `-..__.....----...., `.
                     '                 `  '''---..-''`'              : :  : :
                   .` -                '`.                        `'   `'
                .-` .` '             .`'
            _.-` .-`   '            .
        _.-` _.-`    .' '         .`
(`''--'' _.-`      .'  '        .'
 `'----''        .'  .`       .`
               .'  .'     .-'`
             .'   :    .-`
             `. .`   ,`
              .'   .'
             '   .`
            '  .`
            `  '.
            `.___;

Art by: Y. Huang + Lester
"""


def load_mask(file_path):
    if not any(file_path.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
        raise ValueError(f"Only {ALLOWED_EXTENSIONS} files are allowed. Got: {file_path}")
    image = sitk.ReadImage(file_path)
    mask = sitk.GetArrayFromImage(image)       # ndarray shape: (z, y, x)
    spacing_itk = image.GetSpacing()           # tuple: (sx, sy, sz)
    if mask.ndim != 3:
        raise ValueError(f"Mask from {file_path} is not 3D (found shape: {mask.shape}).")
    spacing = tuple(spacing_itk[::-1])         # reorder to (sz, sy, sx) to match mask axes
    return mask, spacing

def find_file(directory, subject, allowed_extensions=ALLOWED_EXTENSIONS):
    for ext in allowed_extensions:
        fp = os.path.join(directory, subject + ext)
        if os.path.exists(fp):
            return fp
    return None

def evaluate_segmentation_performance(pred_dir, gt_dir, subject_list=None, verbose=False):
    # Load subject list if a JSON path was provided
    if isinstance(subject_list, (str, Path)):
        with open(subject_list, "r") as fp:
            subject_list = json.load(fp)["subject_list"]

    # If not provided, infer subjects from pred_dir (strip extensions)
    if subject_list is None:
        subject_set = set()
        for f in os.listdir(pred_dir):
            f_lower = f.lower()
            for ext in ALLOWED_EXTENSIONS:
                if f_lower.endswith(ext):
                    subject_set.add(f[:-len(ext)])
                    break
        subject_list = sorted(subject_set)

    metrics_list = []
    for subj in subject_list:
        # Map "<case>_seg" -> "<case>" for GT lookup, otherwise keep subj
        base_id = subj[:-4] if subj.endswith("_seg") else subj

        pred_file = find_file(pred_dir, subj)
        gt_file   = find_file(gt_dir, base_id)
        print(gt_file)
        if pred_file is None:
            if verbose: print(f"[skip] Prediction file not found for subject {subj}")
            continue
        if gt_file is None:
            if verbose: print(f"[skip] Ground truth file not found for subject {base_id}")
            continue

        try:
            mask_pred, spacing = load_mask(pred_file)  # spacing now in (sz, sy, sx)
            mask_gt,   spacing_gt = load_mask(gt_file)
        except Exception as e:
            if verbose: print(f"[skip] Error loading subject {subj}: {e}")
            continue

        # Sanity: same shape & spacing
        if mask_gt.shape != mask_pred.shape:
            raise ValueError(f"Shape mismatch for subject {subj}: GT {mask_gt.shape} vs Pred {mask_pred.shape}")
        if not np.allclose(spacing_gt, spacing, rtol=0, atol=1e-4):
            raise ValueError(f"Voxel spacing mismatch for {subj}: GT {spacing_gt} vs Pred {spacing}")

        # Ensure GT uint8 (0/1 assumed)
        mask_gt = mask_gt.astype(np.uint8)

        # Ensure prediction is binary
        uniq = np.unique(mask_pred)
        if not (np.array_equal(uniq, [0]) or np.array_equal(uniq, [1]) or np.array_equal(uniq, [0, 1])):
            if len(uniq) == 2 and 0 in uniq:
                if verbose:
                    print(f"[info] {subj}: pred unique {uniq} -> binarize >0")
                mask_pred = (mask_pred > 0).astype(np.uint8)
            else:
                raise ValueError(f"Prediction mask for {subj} is not binary. Unique: {uniq}")
        else:
            mask_pred = mask_pred.astype(np.uint8)

        # Convert to boolean for surface-distance
        mask_gt_b   = mask_gt.astype(bool)
        mask_pred_b = mask_pred.astype(bool)

        # Handle uniform predictions (all 0 or all 1)
        if np.all(mask_pred_b == 0) or np.all(mask_pred_b == 1):
            if verbose:
                print(f"[warn] {subj}: uniform prediction; setting overlap to 0, max distance penalty.")
            # Max possible diagonal distance (rough bound) in mm, using spacing aligned to array axes
            max_distance = float(np.linalg.norm(np.array(mask_gt.shape) * np.array(spacing_gt)))

            voxel_vol = float(np.prod(spacing_gt))
            gt_volume = float(mask_gt.sum())   * voxel_vol
            pred_volume = float(mask_pred.sum()) * voxel_vol  # not forced to 0; reflects actual mask

            subj_metrics = {
                "subject": subj,
                "volumetric_dice": 0.0,
                "surface_dice": 0.0,
                "hausdorff95": max_distance,
                "masd": max_distance,
                "gt_volume": gt_volume,
                "pred_volume": pred_volume,
            }
            metrics_list.append(subj_metrics)
            continue

        # Surface-based metrics
        sd = surface_metrics.compute_surface_distances(mask_gt_b, mask_pred_b, spacing_mm=spacing_gt)
        dice = float(surface_metrics.compute_dice_coefficient(mask_gt_b, mask_pred_b))
        surf_dice = float(surface_metrics.compute_surface_dice_at_tolerance(sd, tolerance_mm=5))
        hausdorff95 = float(surface_metrics.compute_robust_hausdorff(sd, percent=95))
        avg_gt_to_pred, avg_pred_to_gt = surface_metrics.compute_average_surface_distance(sd)
        masd = float((avg_gt_to_pred + avg_pred_to_gt) / 2.0)

        # Volumes (mm^3)
        voxel_vol = float(np.prod(spacing_gt))
        gt_volume = float(mask_gt.sum())   * voxel_vol
        pred_volume = float(mask_pred.sum()) * voxel_vol

        subj_metrics = {
            "subject": subj,
            "volumetric_dice": dice,
            "surface_dice": surf_dice,
            "hausdorff95": hausdorff95,
            "masd": masd,
            "gt_volume": gt_volume,
            "pred_volume": pred_volume,
        }
        metrics_list.append(subj_metrics)

        if verbose:
            print(f"Subject: {subj}")
            print(f"  Volumetric Dice:   {dice:.4f}")
            print(f"  Surface Dice@5mm:  {surf_dice:.4f}")
            print(f"  Hausdorff95 (mm):  {hausdorff95:.4f}")
            print(f"  MASD (mm):         {masd:.4f}")
            print(f"  GT Vol / Pred Vol: {gt_volume:.2f} / {pred_volume:.2f} mm^3")

    if len(metrics_list) == 0:
        raise RuntimeError("No subjects were processed successfully!")

    mean_dice = float(np.mean([m["volumetric_dice"] for m in metrics_list]))
    mean_surf_dice = float(np.mean([m["surface_dice"] for m in metrics_list]))
    mean_hausdorff95 = float(np.mean([m["hausdorff95"] for m in metrics_list]))
    mean_masd = float(np.mean([m["masd"] for m in metrics_list]))

    gt_volumes = np.array([m["gt_volume"] for m in metrics_list], dtype=float)
    pred_volumes = np.array([m["pred_volume"] for m in metrics_list], dtype=float)
    rmse_volume = float(np.sqrt(np.mean((pred_volumes - gt_volumes) ** 2)))

    aggregates = {
        "mean_volumetric_dice": mean_dice,
        "mean_surface_dice": mean_surf_dice,
        "mean_hausdorff95": mean_hausdorff95,
        "mean_masd": mean_masd,
        "tumor_burden_rmse": rmse_volume,
    }
    return {"per_subject": metrics_list, "aggregates": aggregates}

if __name__ == "__main__":
	import argparse
	import json

	parser = argparse.ArgumentParser(description="Evaluate 3D segmentation performance for .mha and .nii.gz masks")
	parser.add_argument("--pred_dir", type=str, required=True,
						help="Directory containing prediction files (.mha or .nii.gz)")
	parser.add_argument("--gt_dir", type=str, required=True,
						help="Directory containing ground truth files (.mha or .nii.gz)")
	parser.add_argument("--subject_list", type=str, default=None,
						help="Optional JSON file with {'subject_list': [...]}, or a comma-separated list of subject IDs")
	parser.add_argument("--save_path", type=str, default=None,
						help="Optional path to save the aggregated metrics as a JSON file")
	parser.add_argument("--verbose", action="store_true",
						help="Enable verbose output")
	args = parser.parse_args()
	print(panther_msg)
	# Process subject list argument.
	subject_list = args.subject_list
	if subject_list is not None:
		if subject_list.endswith(".json"):
			with open(subject_list, "r") as fp:
				subject_list = json.load(fp)["subject_list"]
		else:
			subject_list = [s.strip() for s in subject_list.split(",")]

	results = evaluate_segmentation_performance(args.pred_dir, args.gt_dir,
												  subject_list=subject_list,
												  verbose=args.verbose)
	print("Evaluation Metrics:")
	print(json.dumps(results, indent=4))
	
		# Save the metrics JSON if a path is provided.
	if args.save_path:
		with open(args.save_path, "w") as f:
			json.dump(results, f, indent=4)
		print(f"Metrics saved to {args.save_path}")
	print(panther_msg2)

# sample test command: python evaluate_local.py --pred_dir test/output/images/pancreatic-tumor-segmentation --gt_dir test_labels --verbose