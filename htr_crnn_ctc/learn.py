from typing import Any, Optional, Literal, Union

import pickle

from skimage.color import rgb2gray
from skimage.transform import rotate
from skimage import io
from torch import Tensor, nn, optim, device as Device
from torch.utils.data import DataLoader
# from torch.utils.data.sampler import SubsetRandomSampler, BatchSampler, Sampler
from numpy.typing import NDArray
from matplotlib import pyplot as plt
from torchvision.transforms import Compose

import torch
import numpy as np
import Levenshtein as leven

from htr_crnn_ctc.model import CTCModel
from htr_crnn_ctc.dataloader import CTCDataLoader
from htr_crnn_ctc.types import DecodeMap, Sample

__all__ = [
    "Learner"
]

class Learner:
    opt: Optional[optim.Adam]
    sched: Optional[optim.lr_scheduler.CyclicLR]

    def __init__(self, model: CTCModel, dataloader: CTCDataLoader, decode_map: DecodeMap,
                 save_path: Optional[str] = None,
                 loss_func=nn.CTCLoss(reduction="sum", zero_infinity=True),
                 optimiser=optim.Adam,
                 scheduler=optim.lr_scheduler.CyclicLR) -> None:
        self.train_dl, self.valid_dl = dataloader()
        self.model = model
        self.loss_func = loss_func
        self.optimiser = optimiser
        self.opt = None
        self.scheduler = scheduler
        self.sched = None
        self.save_path = save_path
        self.decode_map = decode_map
        self.best_leven = 1000

    def fit_one_cycle(self, epochs, max_lr, base_lr=None, base_moms=0.8, max_moms=0.9, wd=1e-2):
        if base_lr is None:
            base_lr = max_lr / 10

        total_batches = epochs * len(self.train_dl)
        up_size = np.floor(total_batches * 0.25)
        down_size = np.floor(total_batches * 0.95 - up_size)

        self.opt = self.optimiser(filter(lambda p: p.requires_grad, self.model.parameters()))
        self.opt.defaults["momentum"] = 0.9
        self.opt.param_groups[0]["momentum"] = 0.9
        self.opt.param_groups[0]["weight_decay"] = wd

        self.sched = self.scheduler(
            self.opt,
            max_lr=max_lr,
            base_lr=base_lr,
            base_momentum=base_moms,
            max_momentum=max_moms,
            step_size_up=up_size,
            step_size_down=down_size
        )

        self.opt.param_groups[0]["betas"] = (
            self.opt.param_groups[0]["momentum"],
            self.opt.param_groups[0]["betas"][1]
        )

        self._fit(epochs=epochs, cyclic=True)

    def fit(self, epochs, lr=1e-3, wd=1e-2, betas=(0.9, 0.999)):
        self.opt = self.optimiser(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=wd,
            betas=betas
        )

        self._fit(epochs=epochs, cyclic=False)

    def _fit(self, epochs, cyclic=False):
        assert self.opt is not None
        assert self.sched is not None

        len_train = len(self.train_dl)
        for i in range(1, epochs + 1):
            batch_n = 1
            train_loss = 0
            loss: Any = 0
            train_leven = 0
            len_leven = 0

            for xb, yb, lens in self.train_dl:
                self.model.train()

                print("epoch {}: batch {} out of {} | loss {}".format(i, batch_n, len_train, loss), end="\r", flush=True)

                self.opt.zero_grad()
                out: Tensor = self.model(xb)
                log_probs = out.log_softmax(2).requires_grad_()
                input_lengths = torch.full((xb.size()[0], ), self.model.time_step, dtype=torch.long)
                loss = self.loss_func(log_probs, yb, input_lengths, lens)

                with torch.no_grad():
                    train_loss += loss

                # assert isinstance(loss, Tensor)
                loss.backward()
                self.opt.step()

                if cyclic:
                    if self.sched.last_epoch < self.sched.total_size:
                        self.sched.step()
                        self.opt.param_groups[0]["betas"] = (self.opt.param_groups[0]["momentum"], self.opt.param_groups[0]["betas"][1])

                if batch_n > (len_train - 5):
                    self.model.eval()
                    with torch.no_grad():
                        decoded = self.model.best_path_decode(xb)
                        for j in range(0, len(decoded)):
                            pred_word = decoded[j]
                            actual = yb.cpu().numpy()[0 + sum(lens[:j]): sum(lens[:j]) + lens[j]]
                            train_leven += leven.distance("".join(pred_word.astype(str)), "".join(actual.astype(str)))
                        sum_lens: NDArray = sum(lens)
                        len_leven += sum_lens.item()

                batch_n += 1

            self.model.eval()

            with torch.no_grad():
                valid_loss = 0
                cer = 0
                wer = 0
                leven_dist = 0
                target_lengths = 0
                for xb, yb, lens in self.valid_dl:
                    input_lengths = torch.full((xb.size()[0],), self.model.time_step, dtype=torch.long)
                    valid_loss += self.loss_func(self.model(xb).log_softmax(2), yb, input_lengths, lens)
                    decoded = self.model.best_path_decode(xb)
                    for j in range(0, len(decoded)):
                        pred_word = decoded[j]
                        actual = yb.cpu().numpy()[0 + sum(lens[:j]): sum(lens[:j]) + lens[j]]
                        leven_dist += leven.distance("".join(pred_word.astype(str)), "".join(actual.astype(str)))
                        pred_len, actual_len = len(pred_word), len(actual)
                        mismatch = sum(pred_word[:min(pred_len, actual_len)] != actual[:min(pred_len, actual_len)]) + abs(len(pred_word) - len(actual))
                        cer += mismatch
                        if mismatch > 0:
                            wer += 1
                    sum_lens = sum(lens)
                    target_lengths += sum_lens.item()

            # assert self.valid_dl.batch_sampler is not None and isinstance(self.valid_dl.batch_sampler, BatchSampler) and isinstance(self.valid_dl.batch_sampler.sampler, SubsetRandomSampler)

            print("epoch {}: train loss {} | valid loss {} | CER {} | IER {}\nTRAIN LEVEN {} | VAL LEVEN {}".format(
                i,
                train_loss / len(self.train_dl),
                valid_loss / len(self.valid_dl),
                cer / target_lengths,
                wer / len(self.valid_dl.batch_sampler.sampler.indices), # type: ignore
                train_leven / len_leven,
                leven_dist / target_lengths
            ), end="\n")

            if (leven_dist / target_lengths) < self.best_leven:
                self.save(leven=leven_dist / target_lengths)
                self.best_leven = leven_dist / target_lengths

    def find_lr(self, start_lr, end_lr, wd=1e-2, momentum=0.9, num_interval=200, plot=True):
        # https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
        sd = self.model.state_dict()

        if num_interval < len(self.train_dl):
            num = num_interval
        else:
            num = len(self.train_dl) - 1

        multi = (end_lr / start_lr) ** (1/num)
        lr = start_lr
        self.opt = self.optimiser(filter(lambda p: p.requires_grad, self.model.parameters()))
        self.opt.param_groups[0]["lr"] = lr
        self.opt.param_groups[0]["weight_decay"] = wd
        avg_loss = 0.0
        best_loss = 0.0
        batch_num = 0
        losses = []
        lrs = []

        for xb, yb, lens in self.train_dl:
            batch_num += 1
            print("batch {}".format(batch_num), end="\r", flush=True)
            self.model.train()
            out = self.model(xb)
            log_probs: Tensor = out.log_softmax(2).requires_grad_()
            input_lengths = torch.full((xb.size()[0], ), self.model.time_step, dtype=torch.long)
            loss: Tensor = self.loss_func(log_probs, yb, input_lengths, lens)
            avg_loss = momentum * avg_loss + (1 - momentum) * loss.data.item()
            smoothed_loss = avg_loss / (1 - momentum ** batch_num)

            if batch_num > 1 and smoothed_loss > 4 * best_loss:
                self.model.load_state_dict(sd)
                if plot:
                    plt.semilogx(lrs, losses)
                    plt.show()

                return lrs, losses

            if smoothed_loss < best_loss or batch_num == 1:
                best_loss = smoothed_loss

            losses.append(smoothed_loss)
            lrs.append(lr)
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()
            lr *= multi
            self.opt.param_groups[0]["lr"] = lr

        self.model.load_state_dict(sd)
        if plot:
            plt.semilogx(lrs, losses)
            plt.show()

        return lrs, losses

    def save(self, f="model.pth", inv_f="decode_map.pk", leven=None):
        try:
            if not leven is None:
                f = str(leven * 100).replace(".", "_") + "_" + f
            torch.save(self.model.state_dict(), f=f)
            with open(inv_f, "wb") as f:
                pickle.dump(self.decode_map, f)
        except OSError as e:
            print(e)

    def load(self, f="model.pth", inv_f="decode_map.pk", load_decode=True, keep_LSTM=True, freeze_conv=False):
        try:
            state_dict = torch.load(f)
            if not keep_LSTM:
                del state_dict["rnn.atrous_conv.weight"]
                del state_dict["rnn.atrous_conv.bias"]

            self.model.load_state_dict(state_dict, strict=keep_LSTM)
            self.model.eval()

            if freeze_conv:
                self.model.frozen = []
                self.model.to_freeze = []
                for k in self.model.state_dict().keys():
                    if not "running" in k and not "track" in k:
                        self.model.frozen.append(False)
                        if not "rnn." in k:
                            self.model.to_freeze.append(True)
                        else:
                            self.model.to_freeze.append(False)

            if load_decode:
                with open(inv_f, "rb") as f:
                    self.decode_map = pickle.load(f)
        except OSError as e:
            print(e)

    def predict(self, transforms: Compose, img_path: Optional[str] = None, img_ndarray: Optional[NDArray] = None, show_img=False, dev=Device("cpu")):
        if img_path is not None:
            img = io.imread(img_path)
        elif img_ndarray is not None:
            img = img_ndarray
        else:
            raise ValueError("img_path, img_ndarray should not 'None'")

        if show_img:
            f, ax = plt.subplots(1,1)
            ax.imshow(img, cmap="gray")
            f.set_size_inches(10, 3)
            plt.show()

        img = transforms(Sample(index=0, image=img, text="")).image
        assert isinstance(img, Tensor)
        img = img.unsqueeze(0).to(dev)

        outs = self.model.best_path_decode(img)
        pred = "".join([self.decode_map[letter] for letter in outs[0]])

        print(pred)

        return pred

    def freeze(self):
        for i, p in enumerate(self.model.parameters()):
            if self.model.to_freeze[i]:
                p.requires_grad = False
                self.model.frozen[i] = True

    def unfreeze(self):
        for p in self.model.parameters():
            p.requires_grad = True
            self.model.frozen = [False for i in range(0, len(self.model.frozen))]

    def _batch_predict(self, xb, yb, lens, dataloader, show_img, up_to):
        print(f"single batch prediction of {dataloader} dataset")

        self.model.eval()
        with torch.no_grad():
            outs = self.model.best_path_decode(xb)
            for i in range(len(outs)):
                start = sum(lens[:i])
                end = lens[i].item()
                corr = "".join([self.decode_map[letter.item()] for letter in yb[start:start + end]])
                pred = "".join([self.decode_map[letter] for letter in outs[i]])

                if show_img:
                    img = xb[i, :, :, :].permute(1,2,0).cpu().numpy()
                    img = rgb2gray(img)
                    img = rotate(img, angle=90, clip=False, resize=True)
                    f, ax = plt.subplots(1,1)
                    ax.imshow(img, cmap="gray")
                    f.set_size_inches(10, 3)
                    plt.show()

                print("actual: {}".format(corr))
                print("pred:   {}".format(pred))

                if i + 1 == up_to:
                    break

    def batch_predict(self, dataloader: Union[Literal["train", "valid", "both"], DataLoader] = "valid", show_img=False, up_to=None):
        if isinstance(dataloader, DataLoader):
            xb, yb, lens = next(iter(dataloader))
            self._batch_predict(xb, yb, lens, "custom", show_img, up_to)
        else:
            if dataloader == "train" or dataloader == "both":
                xb, yb, lens = next(iter(self.train_dl))
                self._batch_predict(xb, yb, lens, "train", show_img, up_to)

            if dataloader == "valid" or dataloader == "both":
                xb, yb, lens = next(iter(self.valid_dl))
                self._batch_predict(xb, yb, lens, "valid", show_img, up_to)
