import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='METAformer', help='which model to run')
parser.add_argument('--mode', type=str, default='train', help='debug to run')  # train or test
parser.add_argument('--debug', type=eval, default=False, help='debug to run')  # if not add log
parser.add_argument('--dataset', type=str, choices=['METRLA', 'PEMSBAY', 'PEMS08', 'PEMS04'], default='METRLA', help='which dataset to run')
parser.add_argument('--dataset_dir', type=str, default='../data/METR-LA/processed/', help='which dataset to run')
parser.add_argument('--trainval_ratio', type=float, default=0.8, help='the ratio of training and validation data among the total')
parser.add_argument('--val_ratio', type=float, default=0.125, help='the ratio of validation data among the trainval ratio')
parser.add_argument('--num_nodes', type=int, default=207, help='num_nodes')
parser.add_argument('--seq_len', type=int, default=12, help='input sequence length')
parser.add_argument('--horizon', type=int, default=12, help='output sequence length')
parser.add_argument('--input_dim', type=int, default=1, help='number of input channel')
parser.add_argument('--output_dim', type=int, default=1, help='number of output channel')
parser.add_argument('--time_of_day', type=eval, default=True, help='add time of day')
parser.add_argument('--day_of_week', type=eval, default=True, help='add day of week')
parser.add_argument('--steps_per_day', type=int, default=288, help='steps per day')
parser.add_argument('--num_heads', type=int, default=4, help='num heads')
parser.add_argument('--num_layers', type=int, default=3, help='num layers')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--lamb', type=float, default=0.01, help='lamb value for separate loss')
parser.add_argument('--lamb1', type=float, default=0.01, help='lamb1 value for compact loss')

parser.add_argument('--early_stop', type=eval, default=True, help='early_stop')
parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument('--loss_func', type=str, default="masked_mae", help='loss function')
parser.add_argument('--grad_norm', type=eval, default=False, help='grad norm')
parser.add_argument('--max_grad_norm', type=int, default=5, help='max grad norm')
parser.add_argument("--early_stop_patience", type=int, default=30, help="patience used for early stop")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.001, help="base learning rate")
parser.add_argument("--weight_decay", type=float, default=0.0001, help="weight decay")
parser.add_argument("--milestones", type=eval, default=[20, 30], help="milestones")
parser.add_argument("--lr_decay_rate", type=float, default=0.1, help="lr_decay_rate")
parser.add_argument("--cl_step_size", type=int, default=2500, help="cl_decay_steps")
parser.add_argument('--use_cl', type=eval, default=False, help='which gpu to use')  # MegaCRN
parser.add_argument('--device', type=str, default="cuda:4", help='which gpu to use')
parser.add_argument('--seed', type=int, default=12, help='random seed.')
args = parser.parse_args()