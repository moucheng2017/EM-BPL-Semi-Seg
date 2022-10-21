import torch
import os

from libs.old.Models2DOrthogonal import Unet2DMultiChannel


def average_model_weights(model_path,
                          model_width=24,
                          step_lower=20000,
                          step_upper=100000):

    all_models = os.listdir(model_path)
    all_models.sort()
    all_models = [os.path.join(model_path, model_name) for model_name in all_models]

    model_path = all_models[0]
    final_model = Unet2DMultiChannel(in_ch=1, width=24, output_channels=1)
    final_model.to('cuda')
    checkpoint = torch.load(model_path)
    final_model.load_state_dict(checkpoint['model_state_dict'])
    avg_weights = final_model.state_dict()

    for each_model in all_models[1:]:
        current_model = Unet2DMultiChannel(in_ch=1,
                                           width=model_width,
                                           output_channels=1)
        current_model.to('cuda')
        checkpoint = torch.load(each_model)
        current_model.load_state_dict(checkpoint['model_state_dict'])
        current_weights = current_model.state_dict()
        for key in current_weights:
            avg_weights[key] = avg_weights[key] + current_weights[key]
    for key in avg_weights:
        avg_weights[key] = avg_weights[key] / len(all_models)
    final_model.load_state_dict(avg_weights)
    return final_model


def main(model_path):
    avg_model = average_model_weights(model_path)
    save_name = 'avg_model.pt'
    save_name = os.path.join(model_path, save_name)
    torch.save(avg_model, save_name)
    print('Done')


if __name__ == "__main__":

    model_path = '/home/moucheng/projects_codes/Results/airway/2022_07_04/' \
                 'OrthogonalSup2DSingle_e1_l0.0001_b4_w24_s50000_r0.001_c_False_n_False_t1.0/trained_models/'
    main(model_path)



