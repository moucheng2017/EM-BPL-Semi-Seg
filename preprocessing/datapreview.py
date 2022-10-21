import torchio as tio
import argparse
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def args_parser():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('inputsource', type=str, help='path to source images dir')
    parser.add_argument('--inputmasks', type=str, nargs='+', help='path to each additional mask dir')
    parser.add_argument('--outputdirname', '-o', default='output', type=str, help='path to dir to save the resultant')
    return parser

def main(args):
    # make subject from cases
    # get data paths

    source_paths = sorted(Path(args.inputsource).glob('*.nii.gz'))
    mask_paths = []
    label_names = []
    for mask_dir in args.inputmasks:
        mask_dir_path = Path(mask_dir)
        mask_paths.append(sorted(mask_dir_path.glob('*.nii.gz')))
        label_names.append(mask_dir_path.stem)

    for paths in mask_paths:
        assert len(source_paths) == len(paths)

    subjects = []
    for ii in range(len(source_paths)):
        subdict={'image': tio.ScalarImage(source_paths[ii])}
        for label, paths  in zip(label_names, mask_paths):
            subdict[label] = tio.LabelMap(paths[ii])
        subjects.append(tio.Subject(subdict))

    dataset = tio.SubjectsDataset(subjects)
    print('Dataset size:', len(dataset), 'subjects')

    # set up output dirs
    o_dir = Path(args.outputdirname)
    o_dir.mkdir(parents=True, exist_ok=True)

    # # transform to 512x512 cubic
    # transforms = [
    #     tio.CopyAffine(target='image'),
    #     tio.Resize(target_shape=(512, 512, -1), image_interpolation='bspline')
    # ]
    # dataset = tio.SubjectsDataset(subjects, transform=tio.Compose(transforms))


    # save output
    for subject in tqdm(dataset):
        im = subject.image
    #     im.save(osource_dir/Path(im.path.name))
    #     # im.to_gif(output_path=osource_dir/Path(im.path.stem.split('.')[0]+'.gif'), axis=1, duration=10)
    #
    #     lg = subject.lung
    #     lg.save(olung_dir/Path(lg.path.name))
    #
    #     if args.inputairway:
    #         ay = subject.airway
    #         ay.save(oawy_dir/Path(ay.path.name))

        # save image
        plt.close('all')
        subject.plot(output_path=o_dir/Path(im.path.stem.split('.')[0]+'.png'))

if __name__=='__main__':
    parser = argparse.ArgumentParser('save each dataset source with mask with middle slice preview', parents=[args_parser()])
    args = parser.parse_args()
    main(args)
