import torch
from libs.Metrics import segmentation_scores
from libs import Helpers


def validate_base(val_img,
                  val_lbl,
                  # device,
                  model,
                  classes=2,
                  threshold=0.95):

        if torch.sum(val_lbl) > 10:

            val_img = val_img.to('cuda', dtype=torch.float32)
            val_lbl = val_lbl.to('cuda', dtype=torch.float32)

            # forward pass:
            val_img = val_img.unsqueeze(1)
            outputs_dict = model(val_img)
            val_output = outputs_dict['segmentation']

            val_output = torch.sigmoid(val_output)
            val_output = (val_output >= threshold).float()

            eval_mean_iu_ = segmentation_scores(val_lbl.squeeze(), val_output.squeeze(), classes)
            return eval_mean_iu_

        else:
            return 0.0


def validate(validate_loader,
             # device,
             model,
             no_validate=10,
             full_orthogonal=0):

    model.eval()
    with torch.no_grad():
        if full_orthogonal == 1:
            val_iou_d = []
            val_iou_h = []
            val_iou_w = []
            iterator_val_labelled = iter(validate_loader)
            for i in range(no_validate):
                val_dict = Helpers.get_data_dict(validate_loader, iterator_val_labelled)
                val_iou_d.append(validate_base(val_dict["plane_d"][0], val_dict["plane_d"][1], model))
                val_iou_h.append(validate_base(val_dict["plane_h"][0], val_dict["plane_h"][1], model))
                val_iou_w.append(validate_base(val_dict["plane_w"][0], val_dict["plane_w"][1], model))

            val_iou = sum(val_iou_d)/len(val_iou_d) + sum(val_iou_h) / len(val_iou_h) + sum(val_iou_w) / len(val_iou_w)
            return val_iou / 3.

        else:
            val_iou = []
            iterator_val_labelled = iter(validate_loader)
            for i in range(no_validate):
                val_dict = Helpers.get_data_dict(validate_loader, iterator_val_labelled)
                val_iou.append(validate_base(val_dict["plane"][0], val_dict["plane"][1], model))

            # print(val_iou)

            return sum(val_iou)/len(val_iou)
