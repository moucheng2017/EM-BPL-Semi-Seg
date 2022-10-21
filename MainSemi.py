# basic libs:
import argparse
from pathlib import Path
import torch
import timeit
import shutil
import math

from libs.Train import train_semi
from libs import Helpers


def trainBPL(args):
    # fix a random seed:
    Helpers.reproducibility(args)

    # model intialisation:
    model, model_name = Helpers.network_intialisation(args)

    # resume training:
    if args.resume is True:
        model = torch.load(args.checkpoint_path)

    # put model in the gpu:
    model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.l2)

    # data loaders:
    # data_loaders = Helpers.get_data_simple_wrapper(args)

    # make saving directories:
    writer, saved_model_path = Helpers.make_saving_directories(model_name, args)

    # set up timer:
    start = timeit.default_timer()

    # train data loader:
    data_iterators = Helpers.get_iterators(args)

    # train labelled:
    train_labelled_data_loader = data_iterators.get('train_loader_l')
    iterator_train_labelled = iter(train_labelled_data_loader)

    # train unlabelled:
    train_unlabelled_data_loader = data_iterators.get('train_loader_u')
    iterator_train_unlabelled = iter(train_unlabelled_data_loader)

    # running loop:
    for step in range(args.iterations):

        # ramp up alpha and beta:
        current_alpha = Helpers.ramp_up(args.alpha, args.warmup, step, args.iterations)
        current_beta = Helpers.ramp_up(args.beta, args.warmup, step, args.iterations)

        # put model to training mode:
        model.train()

        # labelled data
        labelled_dict = Helpers.get_data_dict(train_labelled_data_loader, iterator_train_labelled)

        # unlabelled data:
        unlabelled_dict = Helpers.get_data_dict(train_unlabelled_data_loader, iterator_train_unlabelled)

        loss_d = train_semi(labelled_img=labelled_dict["plane_d"][0],
                            labelled_label=labelled_dict["plane_d"][1],
                            unlabelled_img=unlabelled_dict["plane_d"][0],
                            model=model,
                            t=args.temp,
                            prior_mu=args.mu,
                            prior_logsigma=args.sigma,
                            augmentation_cutout=args.cutout)

        loss_h = train_semi(labelled_img=labelled_dict["plane_h"][0],
                            labelled_label=labelled_dict["plane_h"][1],
                            unlabelled_img=unlabelled_dict["plane_h"][0],
                            model=model,
                            t=args.temp,
                            prior_mu=args.mu,
                            prior_logsigma=args.sigma,
                            augmentation_cutout=args.cutout)

        loss_w = train_semi(labelled_img=labelled_dict["plane_w"][0],
                            labelled_label=labelled_dict["plane_w"][1],
                            unlabelled_img=unlabelled_dict["plane_w"][0],
                            model=model,
                            t=args.temp,
                            prior_mu=args.mu,
                            prior_logsigma=args.sigma,
                            augmentation_cutout=args.cutout)

        sup_loss = loss_d.get('supervised loss').get('loss') + loss_h.get('supervised loss').get('loss') + loss_w.get('supervised loss').get('loss')
        sup_loss = sup_loss / 3
        # print(sup_loss)

        pseudo_loss = loss_d.get('pseudo loss').get('loss') + loss_h.get('pseudo loss').get('loss') + loss_w.get('pseudo loss').get('loss')
        pseudo_loss = pseudo_loss / 3
        # print(pseudo_loss)

        kl_loss = loss_d.get('kl loss').get('loss') + loss_h.get('kl loss').get('loss') + loss_w.get('kl loss').get('loss')
        kl_loss = kl_loss / 3
        # print(kl_loss)

        loss = sup_loss + current_alpha*pseudo_loss + current_beta*kl_loss
        # print(loss)

        learnt_threshold = loss_d.get('kl loss').get('threshold unlabelled') + loss_h.get('kl loss').get('threshold unlabelled') + loss_w.get('kl loss').get('threshold unlabelled')
        learnt_threshold = learnt_threshold.mean() / 3
        # learnt_threshold = learnt_threshold / 3

        del labelled_dict

        if loss != 0.0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for param_group in optimizer.param_groups:
            # exponential decay
            param_group["lr"] = args.lr * ((1 - float(step) / args.iterations) ** 0.99)

        print(
            'Step [{}/{}], '
            'lr: {:.4f},'
            'loss d: {:.4f}, '
            'loss h: {:.4f}, '
            'loss w: {:.4f}, '
            'pseudo loss: {:.4f}, '
            'kl loss: {:.4f}, '
            'Threshold: {:.4f}'.format(step + 1,
                                       args.iterations,
                                       optimizer.param_groups[0]["lr"],
                                       loss_d.get('supervised loss').get('loss'),
                                       loss_h.get('supervised loss').get('loss'),
                                       loss_w.get('supervised loss').get('loss'),
                                       current_alpha*pseudo_loss,
                                       current_beta*kl_loss,
                                       learnt_threshold)
        )

        # # # ================================================================== #
        # # #                        TensorboardX Logging                        #
        # # # # ================================================================ #

        writer.add_scalars('loss metrics', {'train seg loss d': loss_d.get('supervised loss').get('loss'),
                                            'train seg loss h': loss_h.get('supervised loss').get('loss'),
                                            'train seg loss w': loss_w.get('supervised loss').get('loss'),
                                            'train seg total loss': sup_loss,
                                            'train pseudo loss': args.alpha*pseudo_loss,
                                            'learnt threshold': learnt_threshold,
                                            'train kl loss': args.beta*kl_loss}, step + 1)

        if step > args.saving_starting and step % args.saving_frequency == 0:
            save_model_name_full = saved_model_path + '/' + model_name + '_' + str(step) + '.pt'
            torch.save(model, save_model_name_full)

        elif step > args.iterations - 50 and step % 2 == 0:
            save_model_name_full = saved_model_path + '/' + model_name + '_' + str(step) + '.pt'
            torch.save(model, save_model_name_full)

    save_model_name_full = saved_model_path + '/' + model_name + '_' + str(args.iterations) + '.pt'
    torch.save(model, save_model_name_full)

    stop = timeit.default_timer()
    training_time = stop - start
    print('Training Time: ', training_time)

    save_path = saved_model_path + '/results'
    Path(save_path).mkdir(parents=True, exist_ok=True)
    print('\nTraining finished and model saved\n')

    # zip all models:
    shutil.make_archive(saved_model_path, 'zip', saved_model_path)
    shutil.rmtree(saved_model_path)


if __name__ == '__main__':
    main()















