import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from network import Network
from metric import valid
from torch.utils.data import Dataset
import numpy as np
import argparse
import random
from loss import ContrastiveLoss
from dataloader import load_data

# MNIST-USPS
# BDGP
# CCV
# Fashion
# Caltech-2V
# Caltech-3V
# Caltech-4V
# Caltech-5V
# Cifar10
# Cifar100
# Prokaryotic
# Synthetic3d
Dataname = 'MNIST-USPS'
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--learning_rate", default=0.0003)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--pre_epochs", default=200)
parser.add_argument("--con_epochs", default=50)
parser.add_argument("--feature_dim", default=64)
parser.add_argument("--high_feature_dim", default=20)
parser.add_argument("--temperature", default=1)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.dataset == "MNIST-USPS":
    args.con_epochs = 50
    seed = 10
if args.dataset == "BDGP":
    args.con_epochs = 10 # 20
    seed = 30
if args.dataset == "CCV":
    args.con_epochs = 50 # 100
    seed = 100
    args.tune_epochs = 200
if args.dataset == "Fashion":
    args.con_epochs = 50 # 100
    seed = 10
if args.dataset == "Caltech-2V":
    args.con_epochs = 100
    seed = 200
    args.tune_epochs = 200
if args.dataset == "Caltech-3V":
    args.con_epochs = 100
    seed = 30
if args.dataset == "Caltech-4V":
    args.con_epochs = 100
    seed = 100
if args.dataset == "Caltech-5V":
    args.con_epochs = 100
    seed = 1000000
if args.dataset == "Cifar10":
    args.con_epochs = 10
    seed = 10
if args.dataset == "Cifar100":
    args.con_epochs = 20
    seed = 10
if args.dataset == "Prokaryotic":
    args.con_epochs = 20
    seed = 10000
if args.dataset == "Synthetic3d":
    args.con_epochs = 100
    seed = 100

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(seed)

dataset, dims, view, data_size, class_num = load_data(args.dataset)
data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

def compute_view_value(rs, H, view):
    N = H.shape[0]
    W = []
    # all features are normalized
    global_sim = torch.matmul(H,H.t())
    for v in range(view):
        view_sim = torch.matmul(rs[v],rs[v].t())
        related_sim = torch.matmul(rs[v],H.t())
        # The implementation of MMD
        W_v = (torch.sum(view_sim) + torch.sum(global_sim) - 2 * torch.sum(related_sim)) / (N*N)
        W.append(torch.exp(-W_v))
    W = torch.stack(W)
    W = W / torch.sum(W)
    return W.squeeze()


def pretrain(epoch):
    tot_loss = 0.
    criterion = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        xrs,_,_,_ = model(xs)
        loss_list = []
        for v in range(view):
            loss_list.append(criterion(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    #print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss/len(data_loader)))

def contrastive_train(epoch):
    tot_loss = 0.
    mse = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        xrs, zs, rs, H = model(xs)
        loss_list = []

        # compute adaptive weights for each view
        with torch.no_grad():
            w = compute_view_value(rs, H, view)

        for v in range(view):
            # Self-weighted contrastive learning loss
            loss_list.append(contrastiveloss(H, rs[v], w[v]))
            # Reconstruction loss
            loss_list.append(mse(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    #print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss/len(data_loader)))

accs = []
nmis = []
purs = []
if not os.path.exists('./models'):
    os.makedirs('./models')
T = 1
for i in range(T):
    print("ROUND:{}".format(i+1))
    setup_seed(seed)
    model = Network(view, dims, args.feature_dim, args.high_feature_dim, device)
    print(model)
    model = model.to(device)
    state = model.state_dict()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    contrastiveloss = ContrastiveLoss(args.batch_size, args.temperature, device).to(device)
    best_acc, best_nmi, best_pur = 0, 0, 0

    epoch = 1
    while epoch <= args.pre_epochs:
        pretrain(epoch)
        epoch += 1
    # acc, nmi, pur = valid(model, device, dataset, view, data_size, class_num, eval_h=True, epoch=epoch)

    while epoch <= args.pre_epochs + args.con_epochs:
        contrastive_train(epoch)
        acc, nmi, pur = valid(model, device, dataset, view, data_size, class_num, eval_h=False, epoch=epoch)

        if acc > best_acc:
            best_acc, best_nmi, best_pur = acc, nmi, pur
            state = model.state_dict()
            torch.save(state, './models/' + args.dataset + '.pth')
        epoch += 1

    # The final result
    print('The best clustering performace: ACC = {:.4f} NMI = {:.4f} PUR={:.4f}'.format(best_acc, best_nmi, best_pur))
