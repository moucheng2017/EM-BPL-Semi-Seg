from pathlib import Path
import torch
import timeit
import shutil

from libs.Train3D import train_semi
from libs import Helpers

from libs.Validate import validate

from arguments import get_args


def main(args):
    # fix a random seed:
    Helpers.reproducibility(args)

    # model intialisation:
    model, model_name = Helpers.network_intialisation(args)

    # resume training:
    if args.checkpoint.resume == 1:
        model = torch.load(args.checkpoint.checkpoint_path)

    # put model in the gpu:
    model.cuda()
    # model_ema.cuda()
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

    # validate labelled:
    # val_labelled_data_loader = data_iterators.get('val_loader_l')

    # train unlabelled:
    train_unlabelled_data_loader = data_iterators.get('train_loader_u')
    iterator_train_unlabelled = iter(train_unlabelled_data_loader)

    # initialisation of best acc tracker
    best_train = 0.0

    # running loop:
    for step in range(args.train.iterations):

        # initialisation of validating acc:
        validate_acc = 0.0

        # ramp up alpha and beta:
        current_alpha = Helpers.ramp_up(args.train.alpha, args.train.warmup, step, args.train.iterations, args.train.warmup_start)
        # current_beta = Helpers.ramp_up(args.beta, args.warmup, step, args.train.iterations, args.warmup_start)

        # put model to training mode:
        model.train()
        # model_ema.train()

        # labelled data
        labelled_dict = Helpers.get_data_dict(train_labelled_data_loader, iterator_train_labelled)

        # unlabelled data:
        unlabelled_dict = Helpers.get_data_dict(train_unlabelled_data_loader, iterator_train_unlabelled)

        loss_ = train_semi(labelled_img=labelled_dict.get('img'),
                           labelled_label=labelled_dict.get('lbl'),
                           unlabelled_img=unlabelled_dict.get('img'),
                           model=model,
                           t=args.train.temp,
                           prior_mu=args.train.mu,
                           learn_threshold=args.train.learn_threshold,
                           flag=args.train.threshold_flag)

        sup_loss = loss_.get('supervised losses').get('loss').mean()
        pseudo_loss = 0.1*current_alpha*loss_.get('pseudo losses').get('loss').mean()
        kl_loss = 0.1*current_alpha*loss_.get('kl losses').get('loss').mean()
        loss = sup_loss + pseudo_loss + kl_loss

        train_iou = loss_.get('supervised losses').get('train iou')

        del labelled_dict

        if sup_loss > 0.0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for param_group in optimizer.param_groups:
                param_group["lr"] = args.train.lr * ((1 - float(step) / args.train.iterations) ** 0.99)

            print(
                'Step [{}/{}], '
                'lr: {:.4f},'
                'train iou: {:.4f},'
                'val iou: {:.4f},'
                'loss: {:.4f}, '
                'pseudo loss: {:.4f}, '
                'kl loss: {:.4f}, '
                'Threshold: {:.4f}'.format(step + 1,
                                           args.train.iterations,
                                           optimizer.param_groups[0]["lr"],
                                           train_iou,
                                           validate_acc,
                                           sup_loss.item(),
                                           pseudo_loss.item(),
                                           kl_loss.item(),
                                           loss_['kl losses']['threshold'])
            )

            # # # ================================================================== #
            # # #                        TensorboardX Logging                        #
            # # # # ================================================================ #

            writer.add_scalars('ious', {'train iu': train_iou}, step + 1)
            writer.add_scalars('loss metrics', {'train seg loss': sup_loss,
                                                'learnt threshold': loss_['kl losses']['threshold'],
                                                'train kl loss': kl_loss.item(),
                                                'train pseudo loss': pseudo_loss}, step + 1)

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












