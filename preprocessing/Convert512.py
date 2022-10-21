import torchio as tio
import argparse
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

def args_parser():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('inputsource', type=str, help='path to dir holding the dataset')
    parser.add_argument('inputlung', type=str, help='path to dir holding the dataset')
    parser.add_argument('--inputairway', '-a', default=None, type=str, help='path to dir holding the dataset')
    parser.add_argument('--outputdirname', '-o', default='output', type=str, help='path to dir to save the resultant')
    parser.add_argument('--savepreview', '-p', action='store_true', help='save preview for each case.')
    return parser

def main(args):
    # make subject from cases
    # get data paths
    isource_dir = Path(args.inputsource)
    ilung_dir = Path(args.inputlung)

    isource_paths = sorted(isource_dir.glob('*.nii.gz'))
    ilung_paths = sorted(ilung_dir.glob('*.nii.gz'))

    assert len(isource_paths) == len(ilung_paths)

    if args.inputairway:
        iawy_dir = Path(args.inputairway)
        iawy_paths = sorted(iawy_dir.glob('*.nii.gz'))
        assert len(isource_paths) == len(iawy_paths)
    else:
        iawy_paths = [None]*len(isource_paths)

    subjects = []
    for (source_path, lung_path, awy_path) in zip(isource_paths, ilung_paths, iawy_paths):

        subdict={
            'image': tio.ScalarImage(source_path),
            'lung': tio.LabelMap(lung_path),
        }
        if args.inputairway:
            subdict['airway'] = tio.LabelMap(awy_path)
        subjects.append(tio.Subject(subdict))

    dataset = tio.SubjectsDataset(subjects)
    print('Dataset size:', len(dataset), 'subjects')

    # set up output dirs
    o_dir = isource_dir.parent / args.outputdirname
    osource_dir = o_dir / isource_dir.stem
    olung_dir = o_dir / ilung_dir.stem
    osource_dir.mkdir(parents=True, exist_ok=True)
    olung_dir.mkdir(parents=True, exist_ok=True)

    if args.inputairway:
        oawy_dir = o_dir / iawy_dir.stem
        oawy_dir.mkdir(parents=True, exist_ok=True)

    if args.savepreview:
        p_dir = o_dir/Path('previews')
        p_dir.mkdir(parents=True, exist_ok=True)

    # transform to 512x512 cubic
    transforms = [
        tio.CopyAffine(target='image'),
        tio.Resize(target_shape=(512, 512, -1), image_interpolation='bspline')
    ]
    dataset = tio.SubjectsDataset(subjects, transform=tio.Compose(transforms))


    # save output
    for subject in tqdm(dataset):
        im = subject.image
        im.save(osource_dir/Path(im.path.name))
        # im.to_gif(output_path=osource_dir/Path(im.path.stem.split('.')[0]+'.gif'), axis=1, duration=10)

        lg = subject.lung
        lg.save(olung_dir/Path(lg.path.name))

        if args.inputairway:
            ay = subject.airway
            ay.save(oawy_dir/Path(ay.path.name))

        # save image
        if args.savepreview:
            plt.close('all')
            subject.plot(output_path=p_dir/Path(im.path.stem.split('.')[0]+'.png'))

if __name__=='__main__':
    parser = argparse.ArgumentParser('resize dataset into 512x512x-1, original source, awy and lung masks must all be .nii.gz', parents=[args_parser()])
    args = parser.parse_args()
    main(args)