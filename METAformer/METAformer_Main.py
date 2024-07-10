import sys

sys.path.append('../')

from datetime import datetime
from lib.utils import *
from lib.data_loader import *
from METAformer_Config import args
from METAformer_Trainer import Trainer
from model.dstman import Model as Network


def load_data(args):
    data_loader = load_dataset_v2(args.dataset_dir, args.batch_size, args.batch_size, args.batch_size)
    scaler = data_loader['scaler']
    return data_loader, scaler


def get_log_dir(model, dataset):
    current_time = datetime.now().strftime('%Y%m%d%H%M%S')
    current_dir = os.path.abspath(os.path.join(os.getcwd(), "../"))
    log_dir = os.path.join(current_dir, 'log', model, dataset, current_time)
    return log_dir


def generate_model_components(args):
    # 1. model
    model = Network(
        num_nodes=207,
        input_dim=1,
        output_dim=1,
        horizon=12,
        rnn_units=64,
        minute_size=288,
        num_layers=5,
        cheb_k=2,
        mem_num=80,
        mem_dim=64
    )
    model = model.to(args.device)
    print_model_parameters(model, only_num=False)
    # 2. loss function
    if args.loss_func == 'masked_mae':
        loss = MaskedMAELoss()
    elif args.loss_func == 'mae':
        loss = torch.nn.L1Loss().to(args.device)
    elif args.loss_func == 'mse':
        loss = torch.nn.MSELoss().to(args.device)
    elif args.loss_func == 'smoothloss':
        loss = torch.nn.SmoothL1Loss().to(args.device)
    elif args.loss_func == 'huberloss':
        loss = torch.nn.HuberLoss().to(args.device)
    else:
        raise ValueError
    # 3. optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=1e-8)
    # 4. learning rate decay
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=args.milestones,
        gamma=args.lr_decay_rate,
        verbose=False
    )
    return model, loss, optimizer, lr_scheduler


def init_seed(seed):
    """
    Disable cudnn to maximize reproducibility
    """
    torch.cuda.cudnn_enabled = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == '__main__':
    init_seed(args.seed)
    set_cpu_num(1)
    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.device[5]))
    else:
        args.device = 'mps'
    # 数据、日志（运行路径、保存模型、输出结果）、模型、优化器、损失函数
    data_loader, scaler = load_data(args)

    args.log_dir = get_log_dir(args.model, args.dataset)
    model, loss, optimizer, lr_scheduler = generate_model_components(args)

    trainer = Trainer(
        args=args,
        data_loader=data_loader,
        scaler=scaler,
        model=model,
        loss=loss,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler
    )
    if args.mode == 'train':
        trainer.train()
    elif args.mode == 'test':
        checkpoint = "../log/METAformer/METRLA/20240405124036/METRLA_METAformer_best_model.pth"
        trainer.test(args, model, data_loader, scaler, trainer.logger, save_path=checkpoint)
    else:
        raise ValueError
