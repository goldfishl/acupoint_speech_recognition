from ast_models import ASTModel
import torch


def save_ast_model(self_supervised_model_path="./models/SSAST-Base-Frame-400.pth",
                fine_tuned_model_path='./models/best_audio_model.pth', 
                model_path='./models/audio_model.pth'):
    # description: save the model
    # parameter:
    #   self_supervised_model_path: supervised model in the paper
    #   fine_tuned_model_path: fine tuned model that train by myself
    #   model_path: the path to save the model

    # the fucking AST init arguments
    ## change these arguments if you train a model with different arguments
    fshape = 128
    tshape = 2
    fstride = 128
    tstride = 1
    head_lr = 1
    target_length = 420
    num_mel_bins = 128
    model_size = "base"

    # firstly, must load the supervised model
    model = ASTModel(label_dim=418, fshape=fshape, tshape=tshape, fstride=fstride,
                    tstride=tstride,input_fdim=num_mel_bins, input_tdim=target_length,
                    model_size=model_size, pretrain_stage=False,load_pretrained_mdl_path=self_supervised_model_path)

    ## then load the fine tuned model
    sd = torch.load(fine_tuned_model_path)
    if not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)
    model.load_state_dict(sd, strict=False)

    ## save the model
    torch.save(model, model_path)


if __name__ == "__main__":
    save_ast_model()