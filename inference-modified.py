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

"""

The following is the inference script for the baseline algorithm for Task 2 of the PANTHER challenge.

It is meant to run within a container.

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./do_save.sh

Any container that shows the same behaviour will do, this is purely an example of how one COULD do it.

Reference the documentation to get details on the runtime environment on the GC platform:
https://grand-challenge.org/documentation/runtime-environment/
"""

from pathlib import Path
import time
from mrsegmentator import inference
import glob
import SimpleITK as sitk
import numpy as np
import os
import subprocess
import shutil
from data_utils import *

from evalutils import SegmentationAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)
import warnings
warnings.filterwarnings("ignore")
class PancreaticTumorSegmentationContainer(SegmentationAlgorithm):
    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )
        # input / output paths for nnUNet
        self.nnunet_input_dir = Path("/opt/algorithm/nnunet/input")
        self.nnunet_output_dir = Path("/opt/algorithm/nnunet/output")
        self.nnunet_model_dir = Path("/opt/algorithm/nnunet/nnUNet_results")
        self.mrsegmentator_input_dir = Path("/opt/algorithm/mrsegmentator/input")
        self.mrsegmentator_output_dir = Path("/opt/algorithm/mrsegmentator/output")

        # input / output paths for predictions-model
        folders_with_mri = [folder for folder in os.listdir("/input/images") if "mri" in folder.lower()]
        if len(folders_with_mri) == 1:
            mr_ip_dir_name = folders_with_mri[0]
            print("Folder containing eval image", mr_ip_dir_name)
        else:
            print("Error: Expected one folder containing 'mri', but found", len(folders_with_mri))
            mr_ip_dir_name = 'abdominal-t2-mri' #default value
        
        self.mr_ip_dir = Path(f"/input/images/{mr_ip_dir_name}") #abdominal-t2-mri
        self.output_dir = Path("/output")
        self.output_dir_images = Path(os.path.join(self.output_dir, "images"))
        self.output_dir_seg_mask = Path(os.path.join(self.output_dir_images, "pancreatic-tumor-segmentation"))
        self.segmentation_mask = self.output_dir_seg_mask # / "tumor_seg.mha"
        self.weights_path = Path("/opt/ml/model") #weights can be uploaded as a separate tarball to Grand Challenge (Algorithm > Models). The resources will be extracted to this path at runtime
        self.mrsegmentator_weights = "/opt/ml/model/weights"
        os.environ["MRSEG_WEIGHTS_PATH"] = self.mrsegmentator_weights

        # ensure required folders exist
        self.nnunet_input_dir.mkdir(exist_ok=True, parents=True) #not used in the current implementation
        self.nnunet_output_dir.mkdir(exist_ok=True, parents=True)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.output_dir_seg_mask.mkdir(exist_ok=True, parents=True)
        self.mrsegmentator_input_dir.mkdir(exist_ok=True, parents=True)
        self.mrsegmentator_output_dir.mkdir(exist_ok=True, parents=True)

        # Check if any .mha files were found
        mha_files = sorted(glob.glob(os.path.join(self.mr_ip_dir, "*.mha")))
        if mha_files:
            self.mha_files = mha_files           # list of all .mha
        else:
            self.mha_files = []
            print("No mha images found in input directory")

    # --- replace your run() entirely ---
    def run(self):
        """
        Load T2 MRI volumes and generate tumor segmentations (batch over all .mha in folder).
        """
        _show_torch_cuda_info()
        task = "Dataset391_PANTHER_ROI"
    
        # Stage nnU-Net checkpoints once
        self.move_checkpoints(self.weights_path, folds="0")
    
        if not self.mha_files:
            raise RuntimeError("No .mha found under /input/images/<...mri...>/")
    
        # (Optional) clean nnUNet & MRSeg inputs/outputs from previous runs
        for p in [self.nnunet_input_dir, self.nnunet_output_dir,
                  self.mrsegmentator_input_dir, self.mrsegmentator_output_dir]:
            for f in p.glob("*"):
                try:
                    f.unlink()
                except IsADirectoryError:
                    shutil.rmtree(f, ignore_errors=True)
    
        for src_path in self.mha_files:
            case_id = Path(src_path).stem       # e.g., "patient001"
            print(f"\n=== Processing {case_id} ===")
            t0 = time.perf_counter()
    
            # 1) read image
            itk_image = sitk.ReadImage(src_path)
            print(f"Original image shape: {itk_image.GetSize()}, spacing: {itk_image.GetSpacing()}")
    
            # 2) resample & write MRSegmentator input with _0000 suffix
            low_res_image = resample_img(itk_image)
            mrseg_in = self.mrsegmentator_input_dir / f"{case_id}_0000.nii.gz"
            sitk.WriteImage(low_res_image, str(mrseg_in))
    
            # 3) run MRSegmentator (writes <case_id>_0000_seg.nii.gz in output dir)
            print("Input image for mrsegmentator:", os.listdir(self.mrsegmentator_input_dir))
            inference.infer([str(mrseg_in)], self.mrsegmentator_output_dir, [0])
    
            mrseg_out = self.mrsegmentator_output_dir / f"{case_id}_0000_seg.nii.gz"
            if not mrseg_out.exists():
                raise FileNotFoundError(f"MRSegmentator output not found: {mrseg_out}")
    
            # keep only pancreas==7
            mrseg_mask = sitk.ReadImage(str(mrseg_out))
            mrseg_mask_array = sitk.GetArrayFromImage(mrseg_mask)
            pancreas_mask = (mrseg_mask_array == 7).astype(np.uint8)
            pancreas_mask_itk = sitk.GetImageFromArray(pancreas_mask)
            pancreas_mask_itk.CopyInformation(mrseg_mask)
            print(f"Pancreas mask unique values: {np.unique(pancreas_mask)}, mask shape: {pancreas_mask.shape}")
    
            # 4) crop original to pancreas ROI
            margins = [70, 70, 70]
            cropped_mri, crop_coordinates = CropPancreasROI(itk_image, pancreas_mask_itk, margins)
    
            # nnU-Net input filename must end with _0000
            nnunet_case_in = self.nnunet_input_dir / f"{case_id}_0000.mha"
            sitk.WriteImage(cropped_mri, str(nnunet_case_in))
    
            # 5) nnU-Net predict (writes <case_id>.mha in output dir)
            print(f"Input image nnunet: {os.listdir(self.nnunet_input_dir)}")
            print(f"Input dir:{self.nnunet_input_dir}, output dir:{self.nnunet_output_dir}, task:{task}, folds:0")
            self.predict(
                input_dir=self.nnunet_input_dir,
                output_dir=self.nnunet_output_dir,
                task=task,
                folds="0",
            )
    
            nnunet_out = self.nnunet_output_dir / f"{case_id}.mha"
            if not nnunet_out.exists():
                # nnU-Net sometimes exports .nii.gz depending on config; fallback check:
                alt = self.nnunet_output_dir / f"{case_id}.nii.gz"
                if alt.exists():
                    nnunet_out = alt
                else:
                    raise FileNotFoundError(f"nnU-Net output not found for {case_id}: {nnunet_out}")
    
            print(f"nnUNet output shape: {sitk.ReadImage(str(nnunet_out)).GetSize()}")
    
            # 6) restore to full size & save final prediction per case
            tumor_mask_cropped = sitk.ReadImage(str(nnunet_out))
            final_tumor_mask = restore_to_full_size(tumor_mask_cropped, itk_image, crop_coordinates)
    
            out_file = self.output_dir_seg_mask / f"{case_id}_seg.mha"
            sitk.WriteImage(final_tumor_mask, str(out_file))
            print(f"Wrote: {out_file}")
    
            dt = time.perf_counter() - t0
            print(f"Case {case_id} - prediction time: {dt:.2f}s")

        
    def predict(self, input_dir, output_dir, task="Dataset391_PANTHER_ROI", trainer="nnUNetTrainer_100epochs",
                    configuration="3d_fullres", checkpoint="checkpoint_best.pth", folds="0"):  ####
            """
            Use trained nnUNet network to generate segmentation masks
            """

            # Set environment variables
            os.environ['nnUNet_results'] = str(self.nnunet_model_dir)

            # Run prediction script
            cmd = [
                'nnUNetv2_predict',
                '-d', task,
                '-i', str(input_dir),
                '-o', str(output_dir),
                '-c', configuration,
                '-tr', trainer,
                '--disable_progress_bar',
                '--continue_prediction'
            ]

            if folds:
                cmd.append('-f')
                # If folds is a string and contains a comma, split it; otherwise, wrap it in a list.
                fold_list = folds.split(',') if isinstance(folds, str) and ',' in folds else [folds]
                cmd.extend(fold_list)

            if checkpoint:
                cmd.append('-chk')
                cmd.append(str(checkpoint))

            cmd_str = " ".join(cmd)
            print(f"Running command: {cmd_str}")
            subprocess.check_call(cmd_str, shell=True)

    def move_checkpoints(self, source_dir, folds="0", trainer="nnUNetTrainer_100epochs", task="Dataset391_PANTHER_ROI"):  ####
        """
        Move nnUNet checkpoints to nnUNet_results directory.
        """
        # Create the top-level destination directory if it doesn't exist
        os.makedirs(self.nnunet_model_dir, exist_ok=True)
        print(os.listdir(source_dir))
        task_name = task.split("_")[1]
        
        # Determine fold list, supporting both a comma-separated string or a single value.
        fold_list = folds.split(',') if isinstance(folds, str) and ',' in folds else [str(folds)]
        
        # Move the checkpoints
        for fold in fold_list:
            source_path = os.path.join(source_dir, f"checkpoint_best_{task_name}_fold_{fold}.pth") ####
            destination_path = os.path.join(self.nnunet_model_dir, task, f"{trainer}__nnUNetPlans__3d_fullres", f"fold_{fold}", "checkpoint_best.pth")   ####
            # Create the destination directory if it doesn't exist
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            # Move the file
            try:
                shutil.copyfile(source_path, destination_path)
                print(f"Copied checkpoint for fold {fold} to {destination_path}")
            except FileNotFoundError:
                print(f"Source file not found: {source_path}")
            except Exception as e:
                print(f"Error moving checkpoint for fold {fold}: {e}")


def _show_torch_cuda_info():
    import torch

    print("=+=" * 10)
    print(torch.__version__)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    PancreaticTumorSegmentationContainer().run()
