import torch
import timeit
from libs.Train3D import train_semi, train_sup
from libs import Helpers
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
    train_labelled_data_loader = data_iterators['train_loader_l']
    iterator_train_labelled = iter(train_labelled_data_loader)

    # validate labelled:
    # val_labelled_data_loader = data_iterators.get('val_loader_l')

    if args.train.batch_u > 0:
        # train unlabelled:
        train_unlabelled_data_loader = data_iterators['train_loader_u']
        iterator_train_unlabelled = iter(train_unlabelled_data_loader)
    else:
        pass

    # initialisation of best acc tracker
    best_train = 0.0

    # running loop:
    for step in range(args.train.iterations):

        # ramp up alpha and beta:
        current_alpha = Helpers.ramp_up(args.train.alpha, args.train.warmup, step, args.train.iterations, args.train.warmup_start)

        # put model to training mode:
        model.train()

        # labelled data
        labelled_dict = Helpers.get_data_dict(train_labelled_data_loader, iterator_train_labelled)

        if args.train.batch_u > 0:
            # unlabelled data:
            unlabelled_dict = Helpers.get_data_dict(train_unlabelled_data_loader,
                                                    iterator_train_unlabelled)

            loss_ = train_semi(labelled_img=labelled_dict['img'],
                               labelled_label=labelled_dict['lbl'],
                               unlabelled_img=unlabelled_dict['img'],
                               model=model,
                               t=args.train.temp,
                               pri_mu=args.train.pri_mu,
                               pri_std=args.train.pri_std,
                               flag_post_mu=args.train.flag_post_mu,
                               flag_post_std=args.train.flag_post_std,
                               flag_pri_mu=args.train.flag_pri_mu,
                               flag_pri_std=args.train.flag_pri_std)

            sup_loss = loss_['supervised losses']['loss'].mean()
            pseudo_loss = args.train.beta * current_alpha * loss_['pseudo losses']['loss'].mean()

            if (pseudo_loss > 0.0) and (1.0 > loss_['kl losses']['threshold'] > args.train.conf_lower):
                kl_loss = 0.1 * current_alpha*loss_['kl losses']['loss'].mean()
                loss = sup_loss + pseudo_loss + kl_loss
            else:
                kl_loss = torch.zeros(1).cuda()
                pseudo_loss = torch.zeros(1).cuda()
                loss = sup_loss

        else:
            loss_ = train_sup(labelled_img=labelled_dict['img'],
                              labelled_label=labelled_dict['lbl'],
                              model=model,
                              t=args.train.temp)

            sup_loss = loss_['supervised losses']['loss'].mean()
            loss = sup_loss

        train_iou = loss_['supervised losses']['train iou']

        del labelled_dict

        if sup_loss > 0.0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for param_group in optimizer.param_groups:
                param_group["lr"] = args.train.lr * ((1 - float(step) / args.train.iterations) ** 0.99)

            if args.train.batch_u > 0:
                print(
                    'Step [{}/{}], '
                    'lr: {:.4f},'
                    'iou: {:.4f},'
                    'sup loss: {:.4f}, '
                    'pse loss: {:.4f}, '
                    'kl loss: {:.4f}, '
                    'Threshold: {:.4f}, '
                    'prob l: {:.4f}, '
                    'prob u: {:.4f}'.format(step + 1,
                                            args.train.iterations,
                                            optimizer.param_groups[0]["lr"],
                                            train_iou,
                                            sup_loss.item(),
                                            pseudo_loss.item(),
                                            kl_loss.item(),
                                            loss_['kl losses']['threshold'],
                                            loss_['supervised losses']['prob'],
                                            loss_['pseudo losses']['prob']
                                            ))

                writer.add_scalars('ious', {'train iu': train_iou}, step + 1)

                writer.add_scalars('loss metrics', {'train seg loss': sup_loss,
                                                    'train kl loss': kl_loss.item(),
                                                    'train pseudo loss': pseudo_loss}, step + 1)

                writer.add_scalars('probabilities', {'learnt threshold': loss_['kl losses']['threshold'],
                                                     'prob mean labelled': loss_['supervised losses']['prob'],
                                                     'prob mean unlabelled': loss_['pseudo losses']['prob']}, step + 1)

            else:
                print(
                    'Step [{}/{}], '
                    'lr: {:.4f},'
                    'iou: {:.4f},'
                    'sup loss: {:.4f}, '.format(step + 1,
                                                args.train.iterations,
                                                optimizer.param_groups[0]["lr"],
                                                train_iou,
                                                loss))

                writer.add_scalars('loss metrics', {'train loss': loss.item()}, step + 1)
                writer.add_scalars('ious', {'train iu': train_iou}, step + 1)
                writer.add_scalars('probabilities', {'prob mean labelled': loss_['supervised losses']['prob']}, step + 1)

        else:
            pass

        save_model_name_full = saved_model_path + '/' + model_name + '_current.pt'
        torch.save(model, save_model_name_full)

        if step % 1000 == 0 and step > 0:
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












