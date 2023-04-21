import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
import torch

from solvers.tfjnd_solver import TFJNDSolver
from configs.config import get_config
import utils


def main(config):
    # for fast training.
    torch.backends.cudnn.benchmark = True

    solver = TFJNDSolver(config)

    epoch_max = config.epoch_max
    epoch_val = config.epoch_val
    epoch_save = config.epoch_save

    for epoch in range(solver.epoch_start + 1, epoch_max + 1):
        t_epoch_start = solver.timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]

        train_loss, preds, labels = solver.train()
        log_info.append('train| loss: {:.4f}'.format(train_loss))

        solver.save_checkpoint(epoch, epoch_save)

        with torch.no_grad():
            plcc, srocc, krocc = solver.qa_val()

            log_info.append('plcc: {:.4f}'.format(plcc))
            log_info.append('srocc: {:.4f}'.format(srocc))
            log_info.append('krocc: {:.4f}'.format(krocc))

        t = solver.timer.t()
        prog = (epoch - solver.epoch_start + 1) / (epoch_max - solver.epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

        solver.log(', '.join(log_info))
        if (epoch_val is not None) and (epoch % epoch_val == 0):
            with torch.no_grad():
                solver.val(epoch)


if __name__ == '__main__':
    config = get_config()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if config.multiple_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_ids
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id

    if torch.cuda.is_available():
        print('Found [{}] gpu'.format(torch.cuda.device_count()))
        gpu_id = torch.cuda.current_device()
        print('gpu id: [{}], device name: [{}]'.format(gpu_id, torch.cuda.get_device_name(gpu_id)))
    main(config)