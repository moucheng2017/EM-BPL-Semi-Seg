# pip install git+https://github.com/JoHof/lungmask
import os

from lungmask import mask
import SimpleITK as sitk
from pathlib import Path
import argparse

def args_parser():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('inputsource', type=str, help='path to dir holding the dataset')
    parser.add_argument('outputdir', default='lung', type=str, help='path to dir to save the resultant')
    return parser

def main(args):
    # get all lung rawimage paths
    images_dir = Path(args.inputsource)
    image_paths = sorted(images_dir.glob('*.nii.gz'))

    label_dir = Path(args.outputdir)
    label_dir.mkdir(exist_ok=True)

    # load rawimage
    for i, path in enumerate(image_paths):
        print(f'starting {i} of {len(image_paths)}')
        try:
            input_image = sitk.ReadImage(str(path))
            # apply model
            result = mask.apply(input_image)
            # make binary segmentation
            result_processed = result.copy()
            result_processed[result > 1] = 1
            # convert to itk
            result_itk = sitk.GetImageFromArray(result_processed)
            result_itk.CopyInformation(input_image)
            # save result
            seg_path_name = label_dir / (path.stem[:-7] + '_lunglabel.nii.gz')
            sitk.WriteImage(result_itk, str(seg_path_name))
        except RuntimeError:
            print(f'FAILED :{path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generate lung mask using UNET model trained on diverse data', parents=[args_parser()])
    # args = parser.parse_args()
    # main(args)
    main_folder = '/home/moucheng/projects_data/Pulmonary_data/Leuven familial fibrosis/nifti'
    lungmask_folder = '/home/moucheng/projects_data/Pulmonary_data/Leuven familial fibrosis/lungmask'
    all_cases = os.listdir(main_folder)
    # all_cases = [os.path.join(main_folder, case_no) for case_no in all_cases]
    for each_case in all_cases:
        case_source_path = os.path.join(main_folder, each_case)
        store_case_path = os.path.join(lungmask_folder, each_case)
        os.makedirs(store_case_path, exist_ok=True)
        all_files = os.listdir(case_source_path)
        for each_file in all_files:
            nifti_path = os.path.join(case_source_path, each_file)
            input_image = sitk.ReadImage(str(nifti_path))
            # apply model
            result = mask.apply(input_image)
            # make binary segmentation
            result_processed = result.copy()
            result_processed[result > 1] = 1
            # convert to itk
            result_itk = sitk.GetImageFromArray(result_processed)
            result_itk.CopyInformation(input_image)
            # save result
            seg_path_name = store_case_path + '/' + each_file[:-7] + '_lunglabel.nii.gz'
            # seg_path_name = store_case_path / (path.stem[:-4] + '_lunglabel.nii.gz')
            sitk.WriteImage(result_itk, str(seg_path_name))
            print(each_case + ' ' + each_file + ' is done')



