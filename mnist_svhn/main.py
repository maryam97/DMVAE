from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import random
import torch
import os
import numpy as np
from torchvision.datasets import MNIST, SVHN
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from datasets import getPairedDataset
from model import EncoderA, EncoderB, DecoderA, DecoderB
from classifier import MNIST_Classifier, SVHN_Classifier
from util import unpack_data, apply_poe


import sys
sys.path.append('../')
import probtorch
import wandb
# ------------------------------------------------
# training parameters

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=int, default=1, metavar='N',
                        help='run_id')
    parser.add_argument('--run_desc', type=str, default='',
                        help='run_id desc')
    parser.add_argument('--n_shared', type=int, default=10,
                        help='size of the latent embedding of shared')
    parser.add_argument('--n_privateA', type=int, default=1,
                        help='size of the latent embedding of private')
    parser.add_argument('--n_privateB', type=int, default=4,
                        help='size of the latent embedding of private')
    parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                        help='input batch size for training [default: 100]')
    parser.add_argument('--ckpt_epochs', type=int, default=7, metavar='N',
                        help='number of epochs to train [default: 200]')
    parser.add_argument('--epochs', type=int, default=7, metavar='N',
                        help='number of epochs to train [default: 200]')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate [default: 1e-3]')

    parser.add_argument('--label_frac', type=float, default=1.,
                        help='how many labels to use')
    parser.add_argument('--sup_frac', type=float, default=1.,
                        help='supervision ratio')
    parser.add_argument('--lambda_text1', type=float, default=50.,
                        help='multipler for text reconstruction [default: 10]')
    parser.add_argument('--lambda_text2', type=float, default=1.,
                        help='multipler for text reconstruction [default: 10]')
    parser.add_argument('--beta1', type=float, default=1.,
                        help='multipler for TC [default: 10]')
    parser.add_argument('--beta2', type=float, default=1.,
                        help='multipler for TC [default: 10]')
    parser.add_argument('--seed', type=int, default=1, metavar='N',
                        help='random seed for get_paired_data')
    parser.add_argument('--wseed', type=int, default=0, metavar='N',
                        help='random seed for weight')

    parser.add_argument('--ckpt_path', type=str, default='../weights/mnist_svhn_cont',
                        help='save and load path for ckpt')
    parser.add_argument('--use_subset', default=1.0, type=float)
    parser.add_argument('--wandb', action='store_true', default=False, help='wandb')
    parser.add_argument('--wandb_name', type=str, default="CDMVAE", help='wandb name')
    parser.add_argument("--wandb_key",
                        type=str,
                        default="b101bb1b7bc1ac85c436b88c8a809ec31e5aea9f",
                        help="enter your wandb key if you didn't set on your os")  # Maryam's key

    args = parser.parse_args()

# ------------------------------------------------


EPS = 1e-9
CUDA = torch.cuda.is_available()

if CUDA:
    device = 'cuda'
    num_workers = 1
else:
    device = 'cpu'
    num_workers = 0

# added by Maryam
# set seeds
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
# print('torch seed=', torch.seed())
print('torch init seed=', torch.initial_seed())

# path parameters
MODEL_NAME = 'mnist_svhn_cont2-run_id%d-privA%02ddim-privB%02ddim-sh%02ddim-lamb_text1_%s-lamb_text2_%s-beta1%s-beta2%s-seed%s-bs%s-wseed%s' % (
    args.run_id, args.n_privateA, args.n_privateB, args.n_shared, args.lambda_text1,
    args.lambda_text2, args.beta1,
    args.beta2, args.seed,
    args.batch_size, args.wseed)

params = [args.n_privateA, args.n_privateB, args.n_shared, args.lambda_text1, args.lambda_text2, args.beta1, args.beta2]

print('privateA', 'privateB', 'shared', 'lambda_text1', 'lambda_text2', 'beta1', 'beta2')
print(params)

if not os.path.isdir(args.ckpt_path):
    os.makedirs(args.ckpt_path)

if len(args.run_desc) > 1:
    desc_file = os.path.join(args.ckpt_path, 'run_id' + str(args.run_id) + '.txt')
    with open(desc_file, 'w') as outfile:
        outfile.write(args.run_desc)

BETA1 = (1., args.beta1, 1.)
BETA2 = (1., args.beta2, 1.)
# model parameters
NUM_PIXELS = int(28 * 28)
TEMP = 0.66
NUM_SAMPLES = 1
data_path = '../../data/mnist-svhn'

train_mnist_svhn, test_mnist_svhn = getPairedDataset(data_path, 100, cuda=CUDA)
kwargs = {'num_workers': num_workers, 'pin_memory': True} if CUDA else {}
train_loader = DataLoader(train_mnist_svhn, batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = DataLoader(test_mnist_svhn, batch_size=args.batch_size, shuffle=False, **kwargs)

BIAS_TRAIN = (train_loader.dataset.__len__() - 1) / (args.batch_size - 1)
BIAS_TEST = (test_loader.dataset.__len__() - 1) / (args.batch_size - 1)


def cuda_tensors(obj):
    for attr in dir(obj):
        value = getattr(obj, attr)
        if isinstance(value, torch.Tensor):
            setattr(obj, attr, value.cuda())


encA = EncoderA(args.seed, zShared_dim=args.n_shared, zPrivate_dim=args.n_privateA)
decA = DecoderA(args.seed, zShared_dim=args.n_shared, zPrivate_dim=args.n_privateA)
encB = EncoderB(args.seed, zShared_dim=args.n_shared, zPrivate_dim=args.n_privateB)
decB = DecoderB(args.seed, zShared_dim=args.n_shared, zPrivate_dim=args.n_privateB)

# wandb
if args.wandb:
    # wandb.login()
    os.environ['WANDB_API_KEY'] = args.wandb_key
    os.environ['WANDB_CONFIG_DIR'] = "."  # /home/mehgdam1/CDMVAE/ #for docker
    run = wandb.init(project=args.wandb_name)
    wandb.config.update(args)

if CUDA:
    encA.cuda()
    decA.cuda()
    encB.cuda()
    decB.cuda()
    cuda_tensors(encA)
    cuda_tensors(decA)
    cuda_tensors(encB)
    cuda_tensors(decB)

optimizer = torch.optim.Adam(
    list(encB.parameters()) + list(decB.parameters()) + list(encA.parameters()) + list(decA.parameters()),
    lr=args.lr)

mnist_test_loader = torch.utils.data.DataLoader(
    MNIST('../../data/mnist', train=False, download=True, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=False)

svhn_test_loader = torch.utils.data.DataLoader(
    SVHN('../../data/svhn', split='test', download=True, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=False)

mnist_net, svhn_net = MNIST_Classifier(), SVHN_Classifier()
if CUDA:
    mnist_net = mnist_net.cuda()
    svhn_net = svhn_net.cuda()
    # mnist_net.load_state_dict(torch.load('../../data/mnist-svhn/mnist_model.pt'))
    # svhn_net.load_state_dict(torch.load('../../data/mnist-svhn/svhn_model.pt'))
# else:
#     mnist_net.load_state_dict(torch.load('../../data/mnist-svhn/mnist_model.pt', map_location='cpu'))
#     svhn_net.load_state_dict(torch.load('../../data/mnist-svhn/svhn_model.pt', map_location='cpu'))
mnist_net.eval()
svhn_net.eval()

def elbo(q, pA, pB, lamb1=1.0, lamb2=1.0, beta1=(1.0, 1.0, 1.0), beta2=(1.0, 1.0, 1.0), bias=1.0):
    # from each of modality
    reconst_loss_A, kl_A = probtorch.objectives.mws_tcvae.elbo(q, pA, pA['images1_sharedA'],
                                                               latents=['privateA', 'sharedA'], sample_dim=0,
                                                               batch_dim=1,
                                                               beta=beta1, bias=bias)
    
    reconst_loss_B, kl_B = probtorch.objectives.mws_tcvae.elbo(q, pB, pB['images2_sharedB'],
                                                               latents=['privateB', 'sharedB'],
                                                               sample_dim=0, batch_dim=1,
                                                               beta=beta2, bias=bias)
    reconst_loss_A, kl_A = torch.tensor(0), torch.tensor(0)
    reconst_loss_poeA, kl_poeA = probtorch.objectives.mws_tcvae.elbo(q, pA, pA['images1_poe'],
                                                                     latents=['privateA', 'poe'], sample_dim=0,
                                                                     batch_dim=1,
                                                                     beta=beta1, bias=bias)
    reconst_loss_poeB, kl_poeB = probtorch.objectives.mws_tcvae.elbo(q, pB, pB['images2_poe'],
                                                                     latents=['privateB', 'poe'], sample_dim=0,
                                                                     batch_dim=1,
                                                                     beta=beta2, bias=bias)
    reconst_loss_poeA, kl_poeA = torch.tensor(0), torch.tensor(0)
    # # by cross
    reconst_loss_crA, kl_crA = probtorch.objectives.mws_tcvae.elbo(q, pA, pA['images1_sharedB'],
                                                                   latents=['privateA', 'sharedB'], sample_dim=0,
                                                                   batch_dim=1,
                                                                   beta=beta1, bias=bias)
    reconst_loss_crB, kl_crB = probtorch.objectives.mws_tcvae.elbo(q, pB, pB['images2_sharedA'],
                                                                   latents=['privateB', 'sharedA'], sample_dim=0,
                                                                   batch_dim=1,
                                                                   beta=beta2, bias=bias)
    reconst_loss_crA, kl_crA = torch.tensor(0), torch.tensor(0)


    # reconst_loss_crA = torch.tensor(0)
    # reconst_loss_crB = torch.tensor(0)

    # reconst_loss_poeA = torch.tensor(0)
    # reconst_loss_poeB = torch.tensor(0)

    loss = (lamb1 * reconst_loss_A - kl_A) + (lamb2 * reconst_loss_B - kl_B) + \
            (lamb1 * reconst_loss_crA - kl_crA) + (lamb2 * reconst_loss_crB - kl_crB)  + \
    (lamb1 * reconst_loss_poeA - kl_poeA) + (lamb2 * reconst_loss_poeB - kl_poeB)

    # return -loss, [reconst_loss_A, reconst_loss_poeA, reconst_loss_crA], [reconst_loss_B, reconst_loss_poeB,
    #                                                                       reconst_loss_crB]
    return -loss, [reconst_loss_A, reconst_loss_poeA, reconst_loss_crA], [reconst_loss_B, reconst_loss_poeB,
            reconst_loss_crB], [kl_A, kl_poeA, kl_crA], [kl_B, kl_poeB, kl_crB]





# def get_stat():
#     n_samples = 1000
#     random_idx = list(range(len(test_loader) * args.batch_size))
#     random.seed(10)
#     random.shuffle(random_idx)
#
#     min_muPrivateA = []
#     max_muPrivateA = []
#     min_muSharedA = []
#     max_muSharedA = []
#     min_muPrivateB = []
#     max_muPrivateB = []
#     min_muSharedB = []
#     max_muSharedB = []
#
#     for b in range(int(n_samples/100)):
#         fixed_XA = [0] * 100
#         fixed_XB = [0] * 100
#         for i, idx in enumerate(random_idx[b*100: (b+1)*100]):
#             dataT = test_loader.dataset.__getitem__(idx)[0:2]
#             data = unpack_data(dataT, device)
#             fixed_XA[i] = data[0].view(-1, NUM_PIXELS)
#             fixed_XB[i] = data[1]
#             fixed_XA[i] = fixed_XA[i].squeeze(0)
#
#         fixed_XA = torch.stack(fixed_XA, dim=0)
#         fixed_XB = torch.stack(fixed_XB, dim=0)
#
#
#         q = encA(fixed_XA, num_samples=1)
#         q = encB(fixed_XB, num_samples=1, q=q)
#         muPrivateA= q['privateA'].dist.loc.squeeze(0)
#         muSharedA= q['sharedA'].dist.loc.squeeze(0)
#         muPrivateB= q['privateB'].dist.loc.squeeze(0)
#         muSharedB= q['sharedB'].dist.loc.squeeze(0)
#
#         min_muPrivateA.append(muPrivateA.min(dim=0).values)
#         max_muPrivateA.append(muPrivateA.max(dim=0).values)
#         min_muSharedA.append(muSharedA.min(dim=0).values)
#         max_muSharedA.append(muSharedA.max(dim=0).values)
#         min_muPrivateB.append(muPrivateB.min(dim=0).values)
#         max_muPrivateB.append(muPrivateB.max(dim=0).values)
#         min_muSharedB.append(muSharedB.min(dim=0).values)
#         max_muSharedB.append(muSharedB.max(dim=0).values)
#     min_muPrivateA = torch.stack(min_muPrivateA).min(dim=0).values.cpu().detach().numpy()
#     max_muPrivateA = torch.stack(max_muPrivateA).max(dim=0).values.cpu().detach().numpy()
#     min_muSharedA = torch.stack(min_muSharedA).min(dim=0).values.cpu().detach().numpy()
#     max_muSharedA = torch.stack(max_muSharedA).max(dim=0).values.cpu().detach().numpy()
#     min_muPrivateB = torch.stack(min_muPrivateB).min(dim=0).values.cpu().detach().numpy()
#     max_muPrivateB = torch.stack(max_muPrivateB).max(dim=0).values.cpu().detach().numpy()
#     min_muSharedB = torch.stack(min_muSharedB).min(dim=0).values.cpu().detach().numpy()
#     max_muSharedB = torch.stack(max_muSharedB).max(dim=0).values.cpu().detach().numpy()
#
#     return [min_muPrivateA, max_muPrivateA], [min_muSharedA, max_muSharedA], \
# [min_muPrivateB, max_muPrivateB], [min_muSharedB, max_muSharedB]





def cross_acc_prior():
    encA.eval()
    decA.eval()
    encB.eval()
    decB.eval()
    N = 0
    accA= accB = accA_cr = accB_cr = accA_poe = accB_poe = 0
    joint = 0

    # prior for z_private
    p = probtorch.Trace()
    p.normal(loc=torch.zeros((1,args.batch_size, args.n_privateA)),
             scale=torch.ones((1,args.batch_size, args.n_privateA)),
             name='priorA')

    p.normal(loc=torch.zeros((1,args.batch_size, args.n_privateB)),
             scale=torch.ones((1,args.batch_size, args.n_privateB)),
             name='priorB')

    p.normal(loc=torch.zeros((1,args.batch_size, args.n_shared)),
             scale=torch.ones((1,args.batch_size, args.n_shared)),
             name='priorSh')

    num_batches = len(train_loader) + 1
    if args.use_subset < 1.0:
        num_batches_max = max(int(num_batches * args.use_subset), 5)
        print(f"Using {args.use_subset} of the dataset: {num_batches_max}/{num_batches}")
    else:
        num_batches_max = num_batches + 1
    print('num_batches_max=', num_batches_max)
    for i, dataT in enumerate(test_loader):
        if i == num_batches_max: break
        data = unpack_data(dataT, device)
        if data[0].size()[0] == args.batch_size:
            N += 1

            labels = dataT[0][1].cpu().detach().numpy()
            q = encA(data[0].view(-1, NUM_PIXELS), num_samples=NUM_SAMPLES)
            q = encB(data[1], num_samples=NUM_SAMPLES, q=q)

            privA = p['priorA'].dist.sample().to(device)
            privB = p['priorB'].dist.sample().to(device)

            ### own recont ###
            reconA = decA.forward2(torch.cat([privA, q['sharedA'].value], -1)).view(data[0].size()).squeeze(0)
            reconB = decB.forward2(torch.cat([privB, q['sharedB'].value], -1)).squeeze(0)

            ### cross recont ###
            reconA_cross = decA.forward2(torch.cat([privA, q['sharedB'].value], -1)).view(data[0].size()).squeeze(0)
            reconB_cross = decB.forward2(torch.cat([privB, q['sharedA'].value], -1)).squeeze(0)

            ## poe acc ##
            mu_poe, std_poe = apply_poe(CUDA, q['sharedA'].dist.loc, q['sharedA'].dist.scale,
                                                       q['sharedB'].dist.loc, q['sharedB'].dist.scale)
            q.normal(mu_poe,
                     std_poe,
                     name='poe')
            reconA_poe = decA.forward2(torch.cat([privA, q['poe'].value], -1)).view(data[0].size()).squeeze(0)
            reconB_poe = decB.forward2(torch.cat([privB, q['poe'].value], -1)).squeeze(0)


            pred_labelA = mnist_net(reconA)
            pred_labelB = svhn_net(reconB)
            pred_labelA = torch.argmax(pred_labelA, dim=1).cpu().detach().numpy()
            pred_labelB = torch.argmax(pred_labelB, dim=1).cpu().detach().numpy()
            accA += (pred_labelA == labels).sum() / args.batch_size
            accB += (pred_labelB == labels).sum() / args.batch_size

            ### for joint ###
            priorSh = p['priorSh'].dist.sample().to(device)

            reconA_prior = decA.forward2(torch.cat([privA, priorSh], -1)).view(data[0].size()).squeeze(0)
            reconB_prior = decB.forward2(torch.cat([privB, priorSh], -1)).squeeze(0)
            pred_labelA_prior = mnist_net(reconA_prior)
            pred_labelB_prior  = svhn_net(reconB_prior)
            pred_labelA_prior  = torch.argmax(pred_labelA_prior, dim=1).cpu().detach().numpy()
            pred_labelB_prior  = torch.argmax(pred_labelB_prior, dim=1).cpu().detach().numpy()
            joint +=  (pred_labelA_prior == pred_labelB_prior).sum() / args.batch_size


            pred_labelA_cr = mnist_net(reconA_cross)
            pred_labelB_cr = svhn_net(reconB_cross)
            pred_labelA_cr = torch.argmax(pred_labelA_cr, dim=1).cpu().detach().numpy()
            pred_labelB_cr = torch.argmax(pred_labelB_cr, dim=1).cpu().detach().numpy()
            accA_cr += (pred_labelA_cr == labels).sum() / args.batch_size
            accB_cr += (pred_labelB_cr == labels).sum() / args.batch_size

            pred_labelA_poe = mnist_net(reconA_poe)
            pred_labelB_poe = svhn_net(reconB_poe)
            pred_labelA_poe = torch.argmax(pred_labelA_poe, dim=1).cpu().detach().numpy()
            pred_labelB_poe = torch.argmax(pred_labelB_poe, dim=1).cpu().detach().numpy()
            accA_poe += (pred_labelA_poe == labels).sum() / args.batch_size
            accB_poe += (pred_labelB_poe == labels).sum() / args.batch_size

    accA = np.round(accA / N, 4)
    accB = np.round(accB / N, 4)
    joint = np.round(joint / N, 4)

    accA_cr = np.round(accA_cr / N, 4)
    accB_cr = np.round(accB_cr / N, 4)

    accA_poe = np.round(accA_poe / N, 4)
    accB_poe = np.round(accB_poe / N, 4)


    print('------Reported results------')
    print('Test acc A from B:', accA_cr)
    print('Test acc B from A:', accB_cr)
    print('Test joint:', joint)

    print('------own generated acc ------')
    print('Test acc A:', accA)
    print('Test acc B:', accB)

    print('------poe------')
    print('Test acc A from poe:', accA_poe)
    print('Test acc B from poe:', accB_poe)

    if args.wandb:
        wandb.log({'Test acc A from B': accA_cr})
        wandb.log({'Test acc B from A': accB_cr})
        wandb.log({'Test joint_prior': joint})
        wandb.log({'Test joint_prior': joint})
        wandb.log({'Test self acc A': accA})
        wandb.log({'Test self acc B': accB})
        wandb.log({'Test acc A from poe': accA_poe})
        wandb.log({'Test acc B from poe': accB_poe})



def train(encA, decA, encB, decB, optimizer):
    epoch_elbo = 0.0
    epoch_recA = epoch_rec_poeA = epoch_rec_crA = 0.0
    epoch_recB = epoch_rec_poeB = epoch_rec_crB = 0.0
    klA, kl_crA, kl_poeA = 0, 0, 0
    klB, kl_crB, kl_poeB = 0, 0, 0
    accA, accB = 0, 0
    encA.train()
    encB.train()
    decA.train()
    decB.train()
    N = 0
    torch.autograd.set_detect_anomaly(True)
    num_batches = len(train_loader) + 1
    if args.use_subset < 1.0:
        num_batches_max = max(int(num_batches * args.use_subset), 5)
        print(f"Using {args.use_subset} of the dataset: {num_batches_max}/{num_batches}")
    else:
        num_batches_max = num_batches + 1
    print('num_batches_max=', num_batches_max)
    for i, dataT in enumerate(train_loader):
        if i == num_batches_max: break
        data = unpack_data(dataT, device)
        # data0, data1 = paired modalA&B
        # data2, data3 = random modalA&B
        if data[0].size()[0] == args.batch_size:
            N += 1
            images1 = data[0]
            images2 = data[1]

            images1 = images1.view(-1, NUM_PIXELS)

            optimizer.zero_grad()
            # encode
            # print(images.sum())
            q = encA(images1, num_samples=NUM_SAMPLES)
            q = encB(images2, num_samples=NUM_SAMPLES, q=q)

            ## poe ##
            mu_poe, std_poe = apply_poe(CUDA, q['sharedA'].dist.loc, q['sharedA'].dist.scale,
                                                       q['sharedB'].dist.loc, q['sharedB'].dist.scale)
            q.normal(mu_poe,
                     std_poe,
                     name='poe')

            # decode
            pA = decA(images1, {'sharedA': q['sharedA'], 'sharedB': q['sharedB'], 'poe': q['poe']}, q=q,
                      num_samples=NUM_SAMPLES)
            pB = decB(images2, {'sharedA': q['sharedA'], 'sharedB': q['sharedB'], 'poe': q['poe']}, q=q,
                      num_samples=NUM_SAMPLES)



            # pA = decA(images1, {'sharedA': q['sharedA'], 'sharedB': q['sharedB']}, q=q,
            #           num_samples=NUM_SAMPLES)
            # pB = decB(images2, {'sharedA': q['sharedA'], 'sharedB': q['sharedB']}, q=q,
            #           num_samples=NUM_SAMPLES)


            # loss
            loss, recA, recB, klsA, klsB = elbo(q, pA, pB, lamb1=args.lambda_text1, lamb2=args.lambda_text2, beta1=BETA1, beta2=BETA2,
                                    bias=BIAS_TRAIN)

            loss.backward()
            optimizer.step()
            if CUDA:
                loss = loss.cpu()
                recA[0] = recA[0].cpu()
                recB[0] = recB[0].cpu()

            epoch_elbo += loss.item()
            epoch_recA += recA[0].item()
            epoch_recB += recB[0].item()

            labels = dataT[0][1].cpu().detach().numpy()
            # prior for z_private
            # p = probtorch.Trace()
            # p.normal(loc=torch.zeros((1, args.batch_size, args.n_privateA)).to(device),
            #          scale=torch.ones((1, args.batch_size, args.n_privateA)).to(device),
            #          name='priorA')
            #
            # p.normal(loc=torch.zeros((1, args.batch_size, args.n_privateB)).to(device),
            #          scale=torch.ones((1, args.batch_size, args.n_privateB)).to(device),
            #          name='priorB')
            # privA = p['priorA'].dist.sample().to(device)  #### WHY prior???
            # privB = p['priorB'].dist.sample().to(device)
            privA = q['privateA'].value
            privB = q['privateB'].value
            sharedA = q['sharedA'].value
            sharedB = q['sharedB'].value

            reconA = decA.forward2(torch.cat([privA, sharedA], -1)).view(data[0].size()).squeeze(0)
            reconB = decB.forward2(torch.cat([privB, sharedB], -1)).squeeze(0)
            pred_labelA = mnist_net(reconA)
            pred_labelB = svhn_net(reconB)
            pred_labelA = torch.argmax(pred_labelA, dim=1).cpu().detach().numpy()
            pred_labelB = torch.argmax(pred_labelB, dim=1).cpu().detach().numpy()
            accA += (pred_labelA == labels).sum() / args.batch_size
            accB += (pred_labelB == labels).sum() / args.batch_size

            if CUDA:
                for i in range(2):
                    recA[i] = recA[i].cpu()
                    recB[i] = recB[i].cpu()
            epoch_rec_poeA += recA[1].item()
            epoch_rec_crA += recA[2].item()
            epoch_rec_poeB += recB[1].item()
            epoch_rec_crB += recB[2].item()
            klA += klsA[0].item()
            kl_poeA += klsA[1].item()
            kl_crA += klsA[2].item()
            klB += klsB[0].item()
            kl_poeB += klsB[1].item()
            kl_crB += klsB[2].item()

    print('------own generated acc ------')
    print('train acc A:', accA / N)
    print('train acc B:', accB / N)
    print('epoch_elbo=', epoch_elbo / N)
    print('epoch_recA_self=', epoch_recA / N)
    print('epoch_recA_joint=', epoch_rec_poeA / N)
    print('epoch_recA_cross=', epoch_rec_crA / N)
    print('epoch_recB_self=', epoch_recB / N)
    print('epoch_recB_joint=', epoch_rec_poeB / N)
    print('epoch_recB_cross=', epoch_rec_crB / N)
    print('epoch_klA=', klA / N)
    print('epoch_klB=', klB / N)
    print('epoch_klA_cross=', kl_crA / N)
    print('epoch_klB_cross=', kl_crB / N)
    print('epoch_klA_joint=', kl_poeA / N)
    print('epoch_klB_joint=', kl_poeB / N)

    if args.wandb:
        wandb.log({'Train_acc A': accA / N})
        wandb.log({'Train_acc B': accB / N})
        wandb.log({'Training_loss': epoch_elbo / N})
        wandb.log({'Train_reconst_A_self': epoch_recA / N})
        wandb.log({'Train_reconst_B_self': epoch_recB / N})
        wandb.log({'Train_reconst_A_cross': epoch_rec_crA / N})
        wandb.log({'Train_reconst_B_cross': epoch_rec_crB / N})
        wandb.log({'Train_reconst_A_joint': epoch_rec_poeA / N})
        wandb.log({'Train_reconst_B_joint': epoch_rec_poeB / N})
        wandb.log({'Train_kl_A_self': klA / N})
        wandb.log({'Train_kl_B_self': klB / N})
        wandb.log({'Train_kl_A_cross': kl_crA / N})
        wandb.log({'Train_kl_B_cross': kl_crB / N})
        wandb.log({'Train_kl_A_joint': kl_poeA / N})
        wandb.log({'Train_kl_B_joint': kl_poeB / N})

    return epoch_elbo / N, [epoch_recA / N, epoch_rec_poeA / N, epoch_rec_crA / N], [epoch_recB / N,
                                                                                                   epoch_rec_poeB / N,
                                                                                                   epoch_rec_crB / N]


def test(encA, decA, encB, decB, epoch):
    encA.eval()
    decA.eval()
    encB.eval()
    decB.eval()
    epoch_elbo = 0.0
    N = 0
    for i, dataT in enumerate(test_loader):
        data = unpack_data(dataT, device)
        if data[0].size()[0] == args.batch_size:
            N += 1
            images1 = data[0]
            images2 = data[1]
            images1 = images1.view(-1, NUM_PIXELS)

            # encode
            q = encA(images1, num_samples=NUM_SAMPLES)
            q = encB(images2, num_samples=NUM_SAMPLES, q=q)
            pA = decA(images1, {'sharedA': q['sharedA'], 'sharedB': q['sharedB']}, q=q,
                      num_samples=NUM_SAMPLES)
            pB = decB(images2, {'sharedB': q['sharedB'], 'sharedA': q['sharedA']}, q=q,
                      num_samples=NUM_SAMPLES)

            batch_elbo, _, _ = elbo(q, pA, pB, lamb1=args.lambda_text1, lamb2=args.lambda_text2, beta1=BETA1,
                                    beta2=BETA2, bias=BIAS_TEST)

            if CUDA:
                batch_elbo = batch_elbo.cpu()
            epoch_elbo += batch_elbo.item()

    return epoch_elbo / N


# def save_ckpt(e):
#     if not os.path.isdir(args.ckpt_path):
#         os.mkdir(args.ckpt_path)
#     torch.save(encA.state_dict(),
#                '%s/%s-encA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, e))
#     torch.save(decA.state_dict(),
#                '%s/%s-decA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, e))
#     torch.save(encB.state_dict(),
#                '%s/%s-encB_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, e))
#     torch.save(decB.state_dict(),
#                '%s/%s-decB_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, e))




# if args.ckpt_epochs > 0:
#     if CUDA:
#         encA.load_state_dict(torch.load('%s/%s-encA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs)))
#         decA.load_state_dict(torch.load('%s/%s-decA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs)))
#         encB.load_state_dict(torch.load('%s/%s-encB_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs)))
#         decB.load_state_dict(torch.load('%s/%s-decB_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs)))
#     else:
#         encA.load_state_dict(torch.load('%s/%s-encA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs),
#                                         map_location=torch.device('cpu')))
#         decA.load_state_dict(torch.load('%s/%s-decA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs),
#                                         map_location=torch.device('cpu')))
#         encB.load_state_dict(torch.load('%s/%s-encB_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs),
#                                         map_location=torch.device('cpu')))
#         decB.load_state_dict(torch.load('%s/%s-decB_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs),
#                                         map_location=torch.device('cpu')))


for e in range(args.epochs):
    train_start = time.time()
    train_elbo, rec_lossA, rec_lossB = train(encA, decA, encB, decB,
                                                   optimizer)
    train_end = time.time()
    # save_ckpt(e + 1)


    print('[Epoch %d] Train: ELBO %.4e (%ds)' % (
        e, train_elbo, train_end - train_start))
cross_acc_prior()





def shared_latent(data_loader, encA, n_samples):
    torch.manual_seed(1340)
    random.seed(1340)
    fixed_idxs = random.sample(range(len(data_loader.dataset)), n_samples)
    fixed_XA = [0] * n_samples
    fixed_XB = [0] * n_samples
    labels = [0] * n_samples

    for i, idx in enumerate(fixed_idxs):
        dataT = data_loader.dataset[idx]
        data = unpack_data(dataT, device)
        labels[i] = dataT[0][1]
        fixed_XA[i] = data[0].view(-1, NUM_PIXELS).squeeze(0)
        fixed_XB[i] = data[1]

    fixed_XA = torch.stack(fixed_XA, dim=0)
    fixed_XB = torch.stack(fixed_XB, dim=0)
    labels = np.array(labels)

    q = encA(fixed_XA, num_samples=1)
    q = encB(fixed_XB, num_samples=NUM_SAMPLES, q=q)

    ## poe ##
    sharedPoE, _ = apply_poe(CUDA, q['sharedA'].dist.loc, q['sharedA'].dist.scale,
                                               q['sharedB'].dist.loc, q['sharedB'].dist.scale)



    ######################## shared digit id ########################
    sharedA =  q['sharedA'].dist.loc
    sharedB =  q['sharedB'].dist.loc
    shared = torch.cat([sharedA, sharedB, sharedPoE], dim=1).detach().numpy().squeeze(0)

    # total tsne
    tsne = TSNE(n_components=2, random_state=0)
    X_r2 = tsne.fit_transform(shared)

    target_names = np.unique(labels)
    colors = np.array(
        ['burlywood', 'turquoise', 'darkorange', 'blue', 'green', 'gray', 'red', 'black', 'purple', 'pink'])

    fig = plt.figure()
    fig.tight_layout()
    for color, i, target_name in zip(colors, target_names, target_names):
        plt.scatter(X_r2[:n_samples][labels == i, 0], X_r2[:n_samples][labels == i, 1], alpha=0.7, color=color, marker='+', s=50, linewidths=1, label=target_name)
        plt.scatter(X_r2[n_samples:2*n_samples][labels == i, 0], X_r2[n_samples:2*n_samples][labels == i, 1], alpha=0.6, color=color,
                    s=7, linewidths=1, label=target_name)
    plt.show()

    ######################## all images of B ########################
    tsne = TSNE(n_components=2, random_state=0)
    emb = tsne.fit_transform(q['privateB'].dist.loc.detach().numpy().squeeze(0))

    fig, ax = plt.subplots(**{'figsize': (4, 3)})

    ax.scatter(emb[:,0], emb[:,1])
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    for x0, y0, img in zip(emb[:,0], emb[:,1], fixed_XB):
        img = img.detach().numpy().transpose((1,2,0))
        imagebox = OffsetImage(img, zoom=0.2)
        imagebox.image.axes = ax
        ab = AnnotationBbox(imagebox, (x0, y0), frameon=False)
        ax.add_artist(ab)
    plt.axis('off')
    plt.tight_layout()
    plt.show()






# if args.ckpt_epochs == args.epochs:
#     shared_latent(test_loader, encA, 400)
#     cross_acc_prior()

# else:
#     save_ckpt(args.epochs)
