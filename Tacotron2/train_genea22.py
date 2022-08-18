import os
import time
import math
import h5py
import numpy as np

import torch
from common.model import Tacotron2
from common.logger import Tacotron2Logger
from common.hparams import create_hparams
# from common.hparams_small import create_hparams
from torch.utils.data import DataLoader
from common.loss_function import Tacotron2Loss


class SpeechGestureDataset_Genea22(torch.utils.data.Dataset):
    def __init__(self, h5file1, h5file2=None, sequence_length=300, npy_root=".."):
        self.h5 = h5py.File(h5file1, "r")
        self.len = len(self.h5.keys())
        mel_mean = np.load(os.path.join(npy_root, "mel_mean.npy"))
        mel_std = np.load(os.path.join(npy_root, "mel_std.npy"))
        mfcc_mean = np.load(os.path.join(npy_root, "mfcc_mean.npy"))
        mfcc_std = np.load(os.path.join(npy_root, "mfcc_std.npy"))
        prosody_mean = np.load(os.path.join(npy_root, "prosody_mean.npy"))
        prosody_std = np.load(os.path.join(npy_root, "prosody_std.npy"))
        # motion_mean = np.load(os.path.join(npy_root, "expmap_full_mean.npy"))
        # motion_std = np.load(os.path.join(npy_root, "expmap_full_std.npy"))

        ### Normalized audio feature
        self.mel = [(self.h5[str(i)]["audio"]["melspectrogram"][:] - mel_mean) / mel_std
                    for i in range(self.len)]
        # self.mfcc = [(self.h5[str(i)]["audio"]["mfcc"][:] - mfcc_mean) / mfcc_std
        #              for i in range(self.len)]
        self.mfcc = [np.zeros_like((self.h5[str(i)]["audio"]["mfcc"][:] - mfcc_mean) / mfcc_std)
                     for i in range(self.len)]
        # self.prosody = [(self.h5[str(i)]["audio"]["prosody"][:] - prosody_mean) / prosody_std
        #                 for i in range(self.len)]
        self.prosody = [np.zeros_like((self.h5[str(i)]["audio"]["prosody"][:] - prosody_mean) / prosody_std)
                        for i in range(self.len)]

        ### Unnormalized audio feature
        # self.mel = [self.h5[str(i)]["audio"]["melspectrogram"][:] for i in range(self.len)]
        # self.mfcc = [self.h5[str(i)]["audio"]["mfcc"][:] for i in range(self.len)]
        # self.prosody = [self.h5[str(i)]["audio"]["prosody"][:] for i in range(self.len)]

        self.speaker_id = [self.h5[str(i)]["speaker_id"][:] for i in range(len(self.h5.keys()))]
        self.text = [self.h5[str(i)]["text"][:] for i in range(len(self.h5.keys()))]
        self.motion = [self.h5[str(i)]["motion"]["expmap_upper"][:, :]
                       for i in range(len(self.h5.keys()))]
        # self.motion = [self.h5[str(i)]["motion"]["expmap_full"][:, :]
        #                for i in range(len(self.h5.keys()))] # remove the bodyworld positions
        self.h5.close()

        if h5file2 is not None:
            self.h5 = h5py.File(h5file2, "r")
            self.len = len(self.h5.keys())

            ### Normalized audio feature
            self.mel += [(self.h5[str(i)]["audio"]["melspectrogram"][:] - mel_mean) / mel_std
                        for i in range(self.len)]
            # self.mfcc += [(self.h5[str(i)]["audio"]["mfcc"][:] - mfcc_mean) / mfcc_std
            #              for i in range(self.len)]
            self.mfcc += [np.zeros_like((self.h5[str(i)]["audio"]["mfcc"][:] - mfcc_mean) / mfcc_std)
                         for i in range(self.len)]
            # self.prosody += [(self.h5[str(i)]["audio"]["prosody"][:] - prosody_mean) / prosody_std
            #                 for i in range(self.len)]
            self.prosody += [np.zeros_like((self.h5[str(i)]["audio"]["prosody"][:] - prosody_mean) / prosody_std)
                            for i in range(self.len)]

            ### Unnormalized audio feature
            # self.mel += [self.h5[str(i)]["audio"]["melspectrogram"][:] for i in range(self.len)]
            # self.mfcc += [self.h5[str(i)]["audio"]["mfcc"][:] for i in range(self.len)]
            # self.prosody += [self.h5[str(i)]["audio"]["prosody"][:] for i in range(self.len)]

            self.speaker_id += [self.h5[str(i)]["speaker_id"][:] for i in range(len(self.h5.keys()))]
            self.text += [self.h5[str(i)]["text"][:] for i in range(len(self.h5.keys()))]
            self.motion += [self.h5[str(i)]["motion"]["expmap_upper"][:, :]
                           for i in range(len(self.h5.keys()))]
            # self.motion += [self.h5[str(i)]["motion"]["expmap_full"][:, :]
            #                for i in range(len(self.h5.keys()))]  # remove the bodyworld positions
            self.h5.close()

        print("Total clips:", len(self.motion))
        self.mel_dim = mel_mean.shape[0]
        self.mfcc_dim = mfcc_mean.shape[0]
        self.prosody_dim = prosody_mean.shape[0]
        self.audio_dim = mel_mean.shape[0] + mfcc_mean.shape[0] + prosody_mean.shape[0]
        # self.audio_dim = mel_mean.shape[0] + prosody_mean.shape[0]
        self.segment_length = sequence_length

    def __len__(self):
        return len(self.motion)

    def __getitem__(self, idx):
        total_frame_len = self.mel[idx].shape[0]
        start_frame = np.random.randint(0, total_frame_len - self.segment_length)
        end_frame = start_frame + self.segment_length
        mel = self.mel[idx][start_frame:end_frame]
        mfcc = self.mfcc[idx][start_frame:end_frame]
        prosody = self.prosody[idx][start_frame:end_frame]
        # audio = np.concatenate((mel, prosody), axis=-1) # no mfcc setting
        audio = np.concatenate((mel, mfcc, prosody), axis=-1)

        speaker = np.zeros([self.segment_length, 17])
        speaker[:, self.speaker_id[idx]] = 1
        text = self.text[idx][start_frame:end_frame]
        text = np.concatenate((text, speaker), axis=-1)
        textaudio = np.concatenate((audio, text), axis=-1)
        textaudio = torch.FloatTensor(textaudio).transpose(0, 1)

        gesture = self.motion[idx][start_frame:end_frame]
        gesture = torch.FloatTensor(gesture).transpose(0, 1)
        gate = torch.zeros([self.segment_length, ])
        gate[-1] = 1
        length = torch.LongTensor([self.segment_length])
        return textaudio, length, gesture, gate, length


class SpeechGestureDataset_Genea22_ValSequence(SpeechGestureDataset_Genea22):
    def __getitem__(self, idx):
        total_frame_len = self.mel[idx].shape[0]
        mel = self.mel[idx][:]
        mfcc = self.mfcc[idx][:]
        prosody = self.prosody[idx][:]
        # audio = np.concatenate((mel, prosody), axis=-1)
        audio = np.concatenate((mel, mfcc, prosody), axis=-1)

        speaker = np.zeros([total_frame_len, 17])
        speaker[:, self.speaker_id[idx]] = 1
        text = self.text[idx][:]
        text = np.concatenate((text, speaker), axis=-1)
        textaudio = np.concatenate((audio, text), axis=-1)
        textaudio = torch.FloatTensor(textaudio).transpose(0, 1)

        gesture = self.motion[idx][:]
        gesture = torch.FloatTensor(gesture).transpose(0, 1)
        gate = torch.zeros([total_frame_len,])
        gate[-1] = 1
        length = torch.LongTensor([total_frame_len])
        return textaudio, length, gesture, gate, length



class RandomSampler(torch.utils.data.Sampler):
    def __init__(self, min_id, max_id):
        self.min_id = min_id
        self.max_id = max_id
    def __iter__(self):
        while True:
            yield np.random.randint(self.min_id, self.max_id)


class SequentialSampler(torch.utils.data.Sampler):
    def __init__(self, min_id, max_id):
        self.min_id = min_id
        self.max_id = max_id
    def __iter__(self):
        return iter(range(self.min_id, self.max_id))



def prepare_dataloaders(hparams):
    # Get data, data loaders and collate function ready
    print("Loading dataset into memory ...")
    dataset = SpeechGestureDataset_Genea22("../train_v1_unclipped.h5", "../val_v1_unclipped.h5")
    val_dataset = SpeechGestureDataset_Genea22_ValSequence("../val_v1_unclipped.h5")

    train_loader = DataLoader(dataset, num_workers=0,
                              sampler=RandomSampler(0, len(dataset)),
                              batch_size=hparams.batch_size,
                              pin_memory=True,
                              drop_last=False)

    val_loader = DataLoader(val_dataset, num_workers=0,
                            sampler=SequentialSampler(0, len(val_dataset)),
                            batch_size=1,
                            pin_memory=True,
                            drop_last=False)
    return train_loader, val_loader


def prepare_directories_and_logger(output_directory, log_directory, rank=0):
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        logger = Tacotron2Logger(os.path.join(output_directory, log_directory))
    else:
        logger = None
    return logger


def load_model(hparams):
    model = Tacotron2(hparams).cuda()
    return model


def warm_start_model(checkpoint_path, model):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    return model


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(iteration, filepath))
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)


def validate(model, criterion, val_loader, iteration, logger, teacher_prob):
    """Handles all the validation scoring and printing"""
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for i, batch in enumerate(val_loader):
            x, y = model.parse_batch(batch)
            y_pred = model(x, teacher_prob)
            loss = criterion(y_pred, y)
            reduced_val_loss = loss.item()
            val_loss += reduced_val_loss
            print("Iteration {} ValLoss {:.6f}  ".format(i, val_loss/(i+1)), end="\r")
            if i + 1 == 39:
                break
        val_loss = val_loss / (i + 1)

    model.train()
    print("Validation Loss: {:9f}     ".format(val_loss))
    logger.log_validation(val_loss, model, y, y_pred, iteration)


def train(output_directory, log_directory, checkpoint_path, warm_start, n_gpus,
          hparams):

    os.makedirs(os.path.join(hparams.output_directory, "ckpt"), exist_ok=True)
    # os.makedirs(os.path.join(hparams.output_directory, "mels"), exist_ok=True)

    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    model = load_model(hparams)
    learning_rate = hparams.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=hparams.weight_decay)


    criterion = Tacotron2Loss(hparams.mel_weight, hparams.gate_weight, hparams.vel_weight, hparams.pos_weight, hparams.add_l1_losss)
    logger = prepare_directories_and_logger(output_directory, log_directory)

    train_loader, val_loader = prepare_dataloaders(hparams)

    # Load checkpoint if one exists
    iteration = 0
    if checkpoint_path:
        if warm_start: # set to False
            model = warm_start_model(checkpoint_path, model)
        else:
            model, optimizer, _learning_rate, iteration = load_checkpoint(checkpoint_path, model, optimizer)
            if hparams.use_saved_learning_rate:
                learning_rate = _learning_rate
            # iteration += 1  # next iteration is iteration + 1

    reduced_loss = 0.
    duration = 0.
    teacher_prob = 1.
    model.train()

    # ================ MAIN TRAINNIG LOOP! ===================
    for i, batch in enumerate(train_loader):
        start = time.perf_counter()
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

        model.zero_grad()
        x, y = model.parse_batch(batch)
        y_pred = model(x, teacher_prob)

        loss = criterion(y_pred, y)
        reduced_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.grad_clip_thresh)
        optimizer.step()

        iters_from_last_save = iteration % hparams.iters_per_checkpoint + 1
        running_loss = reduced_loss/iters_from_last_save

        if not math.isnan(reduced_loss):
            duration += time.perf_counter() - start
            print("Iteration: {} Loss: {:.6f} Teacher: {:.8f} {:.2f}s/it              "
                  "".format(iteration + 1, running_loss, teacher_prob, duration/iters_from_last_save), end="\r")
            logger.log_training(running_loss, learning_rate, duration/iters_from_last_save, iteration + 1)

        if (iteration + 1) % hparams.iters_per_checkpoint == 0:
            print()
            duration = 0.
            reduced_loss = 0.
            validate(model, criterion, val_loader, iteration + 1, logger, teacher_prob)
            checkpoint_path = os.path.join(output_directory, "ckpt", "checkpoint_{}.pt".format(iteration + 1))
            save_checkpoint(model, optimizer, learning_rate, iteration + 1, checkpoint_path)

        iteration += 1
        # if iteration > 10000:
        #     teacher_prob *= 0.9999
        #     teacher_prob = max(teacher_prob, 0.8)


if __name__ == '__main__':
    hparams = create_hparams()

    torch.cuda.set_device("cuda:{}".format(hparams.device))
    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    train(hparams.output_directory, hparams.log_directory,
          hparams.checkpoint_path, hparams.warm_start, hparams.n_gpus,
          hparams)
