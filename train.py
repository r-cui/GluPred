import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from eval import run_eval
from model.OhioModel import OhioModel
from utils.OhioDataset import prepare_global_data
from utils.opt import Options
from utils.util import save_ckpt


def run(model, data_loader, opt, optimiser=None):
    input_len = opt.left_len
    missing_len = opt.missing_len
    loss_temp = 0
    n = 0

    for i, example in enumerate(data_loader):
        example = example.cuda()
        batch_size, _, _ = example.shape
        n += batch_size
        # training
        if not optimiser is None:
            model.train()
            output = model(example, input_len)

            loss = torch.mean(torch.square(output[:, -missing_len:, 0] - example[:, -missing_len:, 0]))
            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(filter(lambda x: x.requires_grad, model.parameters()), max_norm=1.0)
            optimiser.step()
            loss_temp += loss.item() * batch_size
        # valid / test
        else:
            model.eval()
            output = model(example, input_len)
            loss = torch.mean(torch.square(output[:, -missing_len:, 0] - example[:, -missing_len:, 0]))
            loss_temp += loss.item() * batch_size

    return loss_temp / n


def main(opt):
    torch.manual_seed(opt.seed)

    global_training_set, train_dataset, test_dataset = prepare_global_data(
        data_dir=opt.data_dir, patient_id=opt.patient, example_len=opt.left_len + opt.missing_len, unimodal=opt.unimodal
    )
    global_training_loader = DataLoader(global_training_set, batch_size=opt.batch_size, shuffle=True, pin_memory=True,
                                        drop_last=True)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=True, pin_memory=True, drop_last=True)
    model = OhioModel(
        d_in=train_dataset[0].shape[1],
        num_layers=opt.num_layers,
        d_model=opt.d_model,
        heads=opt.heads,
        d_ff=opt.d_ff,
        dropout=opt.dropout,
        attention_dropout=opt.attention_dropout,
        single_pred=opt.single_pred
    )
    model.cuda()
    model.train()
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))

    optimiser = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=opt.pre_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=opt.decay_factor, patience=opt.decay_patience)

    # global train
    test_score_best = float("inf")
    for ep in range(1, opt.pre_epoch + 1):
        print("\n>>> Globally training ep {} with lr {}".format(ep, optimiser.param_groups[0]['lr']))
        train_loss = run(
            model=model,
            data_loader=global_training_loader if opt.transfer_learning else train_loader,
            opt=opt,
            optimiser=optimiser
        )
        test_loss = run(
            model=model,
            data_loader=test_loader,
            opt=opt,
            optimiser=None
        )
        scheduler.step(test_loss)
        test_score = run_eval(model, test_loader, opt)
        print("   Training loss: {:.4f}\n   Test loss: {:.4f}".format(train_loss, test_loss))
        print("Test score: {:.3f}".format(test_score))
        if test_score < test_score_best:
            print("New best model here.")
            test_score_best = test_score
            save_ckpt(
                obj={"opt": opt,
                     "state_dict": model.state_dict(),
                     "optimiser": optimiser.state_dict(),
                     "scheduler": scheduler.state_dict()},
                opt=opt
            )

    # personal train
    optimiser = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=opt.ft_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=opt.decay_factor, patience=opt.decay_patience)
    print("==================================\n==================================")
    for ep in range(1, opt.epoch + 1):
        print("\n>>> Fine tuning ep {} with lr {}".format(ep, optimiser.param_groups[0]['lr']))
        train_loss = run(
            model=model,
            data_loader=train_loader,
            opt=opt,
            optimiser=optimiser
        )
        test_loss = run(
            model=model,
            data_loader=test_loader,
            opt=opt,
            optimiser=None
        )
        scheduler.step(test_loss)
        test_score = run_eval(model, test_loader, opt)
        print("   Training loss: {:.4f}\n   Test loss: {:.4f}".format(train_loss, test_loss))
        print("Test score: {:.3f}".format(test_score))
        if test_score < test_score_best:
            print("New best model here.")
            test_score_best = test_score
            save_ckpt(
                obj={"opt": opt,
                     "state_dict": model.state_dict(),
                     "optimiser": optimiser.state_dict(),
                     "scheduler": scheduler.state_dict()},
                opt=opt
            )

    print("\nFinished training! Best test score: {:.3f}".format(test_score_best))


if __name__ == '__main__':
    opt = Options().parse()
    main(opt)
