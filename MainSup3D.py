from pathlib import Path
import torch
import timeit
import shutil
import math
import numpy as np

from libs.Train3D import train_sup
from libs import Helpers
from libs.Validate import validate

from arguments import get_args


def main(args):
    # fix a random seed:
    Helpers.reproducibility(args)

    # model intialisation:
    model, model_name = Helpers.network_intialisation(args)

    # resume training:
    if args.checkpoint.resume is True:
        model = torch.load(args.checkpoint.checkpoint_path)

    # put model in the gpu:
    model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.train.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.train.optimizer.weight_decay)

    # make saving directories:
    writer, saved_model_path = Helpers.make_saving_directories(model_name, args)

    # set up timer:
    start = timeit.default_timer()

    # train data loader:
    data_iterators = Helpers.get_iterators(args)

    # train labelled:
    train_labelled_data_loader = data_iterators.get('train_loader_l')
    iterator_train_labelled = iter(train_labelled_data_loader)

    best_train = 0.0

    # running loop:
    for step in range(args.train.iterations):

        # put model to training mode:
        model.train()

        # labelled data
        labelled_data = Helpers.get_data_dict(train_labelled_data_loader, iterator_train_labelled)

        loss_ = train_sup(labelled_img=labelled_data.get('img'),
                          labelled_label=labelled_data.get('lbl'),
                          model=model,
                          t=args.train.temp)

        loss = loss_['supervised losses']['loss'].mean()

        train_iou = loss_.get('supervised losses').get('train iou')

        del labelled_data

        if loss > 0.0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for param_group in optimizer.param_groups:
                # exponential decay
                param_group["lr"] = args.train.lr * ((1 - float(step) / args.train.iterations) ** 0.99)

            # We disabled validation because it slows down the training
            # validate_acc = validate(validate_loader=val_labelled_data_loader,
            #                         model=model,
            #                         no_validate=args.validate_no,
            #                         full_orthogonal=args.full_orthogonal)

            print(
                'Step [{}/{}], '
                'lr: {:.4f},'
                'train iou: {:.4f},'
                'Train loss: {:.4f}, '.format(step + 1,
                                              args.train.iterations,
                                              optimizer.param_groups[0]["lr"],
                                              train_iou,
                                              loss))

            # # # ================================================================== #
            # # #                        TensorboardX Logging                        #
            # # # # ================================================================ #

            writer.add_scalars('loss metrics', {'train loss': loss.item()}, step + 1)
            writer.add_scalars('ious', {'train iu': train_iou}, step + 1)

        save_model_name_full = saved_model_path + '/' + model_name + '_current.pt'
        torch.save(model, save_model_name_full)

        if step % 5000 == 0 and step > 0:
            save_model_name_full = saved_model_path + '/' + model_name + 'step' + str(step) + '.pt'
            torch.save(model, save_model_name_full)

        if train_iou > best_train:
            save_model_name_full = saved_model_path + '/' + model_name + '_best_train.pt'
            torch.save(model, save_model_name_full)
            best_train = max(best_train, train_iou)

    stop = timeit.default_timer()
    training_time = stop - start
    print('Training Time: ', training_time)


if __name__ == "__main__":
    args = get_args()
    main(args=args)

    # completed_log_dir = args.log_dir.replace('in-progress', 'debug' if args.debug else 'completed')
    # os.rename(args.log_dir, completed_log_dir)
    # print(f'Log file has been saved to {completed_log_dir}')









