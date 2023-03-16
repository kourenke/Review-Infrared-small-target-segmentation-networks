import cv2
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from  mpl_toolkits.mplot3d import Axes3D
from thop import profile
import time

def getZ(img,X,Y):
    gray = img[X,Y]
    return gray


def parse_args():
    #
    # Setting parameters
    #
    parser = ArgumentParser(description='Inference of Net')

    #
    # Checkpoint parameters
    #
    parser.add_argument('--pkl-path_FCN', type=str, default=r'./results/merged_FCN_X32_Iter-12000_mIoU-0.6265_fmeasure-0.7704.pkl',
                        help='checkpoint path')
    parser.add_argument('--pkl-path_UNet', type=str, default=r'./results/merged_UNet_X16_Iter-31400_mIoU-0.6473_fmeasure-0.7859.pkl',
                        help='checkpoint path')
    parser.add_argument('--pkl-path_Fusionnet', type=str, default=r'./results/merged_Fusionnet_X16_Iter-17000_mIoU-0.5981_fmeasure-0.7485.pkl',
                        help='checkpoint path')
    parser.add_argument('--pkl-path_DeepLabv3_plus', type=str, default=r'./results/merged_DeepLabv3_plus_X16_Iter-41500_mIoU-0.5615_fmeasure-0.7192.pkl',
                        help='checkpoint path')
    parser.add_argument('--pkl-path_ENet', type=str, default=r'./results/merged_ENet_X8_Iter-    0_mIoU-0.5913_fmeasure-0.7431.pkl',
                        help='checkpoint path')
    parser.add_argument('--pkl-path_BiSeNet', type=str, default=r'./results/merged_BiSeNet_X8_Iter- 6400_mIoU-0.5336_fmeasure-0.6959.pkl',
                        help='checkpoint path')
    parser.add_argument('--pkl-path_DFA', type=str, default=r'./results/merged_DFANet_Iter-14200_mIoU-0.5232_fmeasure-0.6869.pkl',
                        help='checkpoint path')
    parser.add_argument('--pkl-path_MDvsFA', type=str,default=r'./results/merged_MDvsFA_Iter- 9300_mIoU-0.5119_fmeasure-0.6872.pkl',
                        help='checkpoint path')
    parser.add_argument('--pkl-path_ACM', type=str, default=r'./results/merged_ACMUNet_Iter-    0_mIoU-0.5334_fmeasure-0.6957.pkl',
                        help='checkpoint path')
    parser.add_argument('--pkl-path_LSPM', type=str, default=r'./results/merged_LSPM_Iter-28500_mIoU-0.5334_fmeasure-0.6957.pkl',
                        help='checkpoint path')
    parser.add_argument('--pkl-path_DNA', type=str, default=r'./results/merged_DNANet_Iter-109500_mIoU-0.6542_fmeasure-0.7909.pkl',
                        help='checkpoint path')
    parser.add_argument('--pkl-path_AGPC', type=str, default=r'./results/merged_agpcnet_Iter-43500_mIoU-0.6627_fmeasure-0.7943.pkl',
                        help='checkpoint path')
    parser.add_argument('--pkl-path_LW_IRST', type=str, default=r'./results/merged_LW_IRST_ablation_Iter-18400_mIoU-0.6638_fmeasure-0.7979.pkl',
                        help='checkpoint path')

    #
    # Test image parameters
    #
    parser.add_argument('--image-path', type=str, default=r'./data/single/31.png', help='image path')
    parser.add_argument('--base-size', type=int, default=256, help='base')


    args = parser.parse_args()
    return args


def preprocess_image(img):
    means=[0.485, 0.456, 0.406]
    stds=[0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[: , :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = Variable(preprocessed_img, requires_grad = True)
    return input


if __name__ == '__main__':
    args = parse_args()

    # load image
    print('...loading test image: %s' % args.image_path)
    img = cv2.imread(args.image_path, 1)
    img = np.float32(cv2.resize(img, (args.base_size, args.base_size))) / 255
    input = preprocess_image(img)

    # load network
    # _____________________________________________FCN__________________________________________
    print('...load checkpoint_FCN: %s' % args.pkl_path_FCN)
    net_FCN = torch.load(args.pkl_path_FCN, map_location=torch.device('cpu'))
    net_FCN.eval()
    # _____________________________________________UNet__________________________________________
    print('...load checkpoint_UNet: %s' % args.pkl_path_UNet)
    net_UNet = torch.load(args.pkl_path_UNet, map_location=torch.device('cpu'))
    net_UNet.eval()
    # _____________________________________________Fusionnet__________________________________________
    print('...load checkpoint_Fusionnet: %s' % args.pkl_path_Fusionnet)
    net_Fusionnet = torch.load(args.pkl_path_Fusionnet, map_location=torch.device('cpu'))
    net_Fusionnet.eval()
    # _____________________________________________DeepLabv3_plus__________________________________________
    print('...load checkpoint_DeepLabv3_plus: %s' % args.pkl_path_DeepLabv3_plus)
    net_DeepLabv3_plus = torch.load(args.pkl_path_DeepLabv3_plus, map_location=torch.device('cpu'))
    net_DeepLabv3_plus.eval()
    # _____________________________________________ENet__________________________________________
    print('...load checkpoint_ENet: %s' % args.pkl_path_ENet)
    net_ENet = torch.load(args.pkl_path_ENet, map_location=torch.device('cpu'))
    net_ENet.eval()
    # _____________________________________________BiSeNet__________________________________________
    print('...load checkpoint_BiSeNet: %s' % args.pkl_path_BiSeNet)
    net_BiSeNet = torch.load(args.pkl_path_BiSeNet, map_location=torch.device('cpu'))
    net_BiSeNet.eval()
    # _____________________________________________DFA__________________________________________
    print('...load checkpoint_DFA: %s' % args.pkl_path_DFA)
    net_DFA = torch.load(args.pkl_path_DFA, map_location=torch.device('cpu'))
    net_DFA.eval()
    # _____________________________________________MDvsFA__________________________________________
    print('...load checkpoint_MDvsFA: %s' % args.pkl_path_MDvsFA)
    net_MDvsFA = torch.load(args.pkl_path_MDvsFA, map_location=torch.device('cpu'))
    net_MDvsFA.eval()
    # _____________________________________________ACM__________________________________________
    print('...load checkpoint_ACM: %s' % args.pkl_path_ACM)
    net_ACM = torch.load(args.pkl_path_ACM, map_location=torch.device('cpu'))
    net_ACM.eval()
    # _____________________________________________LSPM__________________________________________
    print('...load checkpoint_LSPM: %s' % args.pkl_path_LSPM)
    net_LSPM = torch.load(args.pkl_path_LSPM, map_location=torch.device('cpu'))
    net_LSPM.eval()
    # _____________________________________________DNA__________________________________________
    print('...load checkpoint_DNA: %s' % args.pkl_path_DNA)
    net_DNA = torch.load(args.pkl_path_DNA, map_location=torch.device('cpu'))
    net_DNA.eval()
    # _____________________________________________AGPC__________________________________________
    print('...load checkpoint_AGPC: %s' % args.pkl_path_AGPC)
    net_AGPC = torch.load(args.pkl_path_AGPC, map_location=torch.device('cpu'))
    net_AGPC.eval()
    # _____________________________________________LW_IRST__________________________________________
    print('...load checkpoint_LW_IRST: %s' % args.pkl_path_LW_IRST)
    net_LW_IRST = torch.load(args.pkl_path_LW_IRST, map_location=torch.device('cpu'))
    net_LW_IRST.eval()

    # inference in cpu
    # _____________________________________________FCN_______________________________________________
    print('...inference in progress FCN')
    start = time.perf_counter()
    with torch.no_grad():
        output_FCN = net_FCN(input)
    end = time.perf_counter()
    running_FPS = 1/(end-start)
    print('running_FPS:', running_FPS)
    FLOPs, params = profile(net_FCN, inputs=(input,))
    print('FLOPs=', str(FLOPs/1000000.0) + '{}'.format('M'))
    print('params=', str(params / 1000000.0) + '{}'.format('M'))

    # _____________________________________________UNet_______________________________________________
    print('...inference in progress UNet')
    start = time.perf_counter()
    with torch.no_grad():
        output_UNet = net_UNet(input)
    end = time.perf_counter()
    running_FPS = 1/(end-start)
    print('running_FPS:', running_FPS)
    FLOPs, params = profile(net_UNet, inputs=(input,))
    print('FLOPs=', str(FLOPs/1000000.0) + '{}'.format('M'))
    print('params=', str(params / 1000000.0) + '{}'.format('M'))

    # _____________________________________________Fusionnet_______________________________________________
    print('...inference in progress Fusionnet')
    start = time.perf_counter()
    with torch.no_grad():
        output_Fusionnet = net_Fusionnet(input)
    end = time.perf_counter()
    running_FPS = 1 / (end - start)
    print('running_FPS:', running_FPS)
    FLOPs, params = profile(net_Fusionnet, inputs=(input,))
    print('FLOPs=', str(FLOPs / 1000000.0) + '{}'.format('M'))
    print('params=', str(params / 1000000.0) + '{}'.format('M'))

    # _____________________________________________DeepLabv3_plus_______________________________________________
    print('...inference in progress DeepLabv3_plus')
    start = time.perf_counter()
    with torch.no_grad():
        output_DeepLabv3_plus = net_DeepLabv3_plus(input)
    end = time.perf_counter()
    running_FPS = 1 / (end - start)
    print('running_FPS:', running_FPS)
    FLOPs, params = profile(net_DeepLabv3_plus, inputs=(input,))
    print('FLOPs=', str(FLOPs / 1000000.0) + '{}'.format('M'))
    print('params=', str(params / 1000000.0) + '{}'.format('M'))

    # _____________________________________________ENet_______________________________________________
    print('...inference in progress ENet')
    start = time.perf_counter()
    with torch.no_grad():
        output_ENet = net_ENet(input)
    end = time.perf_counter()
    running_FPS = 1 / (end - start)
    print('running_FPS:', running_FPS)
    FLOPs, params = profile(net_ENet, inputs=(input,))
    print('FLOPs=', str(FLOPs / 1000000.0) + '{}'.format('M'))
    print('params=', str(params / 1000000.0) + '{}'.format('M'))

    # _____________________________________________BiSeNet_______________________________________________
    print('...inference in progress BiSeNet')
    start = time.perf_counter()
    with torch.no_grad():
        output_BiSeNet = net_BiSeNet(input)
    end = time.perf_counter()
    running_FPS = 1 / (end - start)
    print('running_FPS:', running_FPS)
    FLOPs, params = profile(net_BiSeNet, inputs=(input,))
    print('FLOPs=', str(FLOPs / 1000000.0) + '{}'.format('M'))
    print('params=', str(params / 1000000.0) + '{}'.format('M'))

    # _____________________________________________DFA_______________________________________________
    print('...inference in progress DFA')
    start = time.perf_counter()
    with torch.no_grad():
        output_DFA = net_DFA(input)
    end = time.perf_counter()
    running_FPS = 1 / (end - start)
    print('running_FPS:', running_FPS)
    FLOPs, params = profile(net_DFA, inputs=(input,))
    print('FLOPs=', str(FLOPs / 1000000.0) + '{}'.format('M'))
    print('params=', str(params / 1000000.0) + '{}'.format('M'))

    # _____________________________________________MDvsFA_______________________________________________
    print('...inference in progress MDsFA')
    start = time.perf_counter()
    with torch.no_grad():
        output_MDvsFA = net_MDvsFA(input)
    end = time.perf_counter()
    running_FPS = 1 / (end - start)
    print('running_FPS:', running_FPS)
    FLOPs, params = profile(net_MDvsFA, inputs=(input,))
    print('FLOPs=', str(FLOPs / 1000000.0) + '{}'.format('M'))
    print('params=', str(params / 1000000.0) + '{}'.format('M'))
    # _____________________________________________ACM_______________________________________________
    print('...inference in progress ACM')
    start = time.perf_counter()
    with torch.no_grad():
        output_ACM = net_ACM(input)
    end = time.perf_counter()
    running_FPS = 1 / (end - start)
    print('running_FPS:', running_FPS)
    FLOPs, params = profile(net_ACM, inputs=(input,))
    print('FLOPs=', str(FLOPs / 1000000.0) + '{}'.format('M'))
    print('params=', str(params / 1000000.0) + '{}'.format('M'))

    # _____________________________________________LSPM_______________________________________________
    print('...inference in progress LSPM')
    start = time.perf_counter()
    with torch.no_grad():
        output_LSPM = net_LSPM(input)
    end = time.perf_counter()
    running_FPS = 1 / (end - start)
    print('running_FPS:', running_FPS)
    FLOPs, params = profile(net_LSPM, inputs=(input,))
    print('FLOPs=', str(FLOPs / 1000000.0) + '{}'.format('M'))
    print('params=', str(params / 1000000.0) + '{}'.format('M'))

    # _____________________________________________DNA_______________________________________________
    print('...inference in progress DNA')
    start = time.perf_counter()
    with torch.no_grad():
        output_DNA = net_DNA(input)
    end = time.perf_counter()
    running_FPS = 1 / (end - start)
    print('running_FPS:', running_FPS)
    FLOPs, params = profile(net_DNA, inputs=(input,))
    print('FLOPs=', str(FLOPs / 1000000.0) + '{}'.format('M'))
    print('params=', str(params / 1000000.0) + '{}'.format('M'))

    # _____________________________________________AGPC_______________________________________________
    print('...inference in progress AGPC')
    start = time.perf_counter()
    with torch.no_grad():
        output_AGPC = net_AGPC(input)
    end = time.perf_counter()
    running_FPS = 1 / (end - start)
    print('running_FPS:', running_FPS)
    FLOPs, params = profile(net_AGPC, inputs=(input,))
    print('FLOPs=', str(FLOPs / 1000000.0) + '{}'.format('M'))
    print('params=', str(params / 1000000.0) + '{}'.format('M'))

    # _____________________________________________LW_IRST_______________________________________________
    print('...inference in progress LW_IRST')
    start = time.perf_counter()
    with torch.no_grad():
        output_LW_IRST = net_LW_IRST(input)
    end = time.perf_counter()
    running_FPS = 1 / (end - start)
    print('running_FPS:', running_FPS)
    FLOPs, params = profile(net_LW_IRST, inputs=(input,))
    print('FLOPs=', str(FLOPs / 1000000.0) + '{}'.format('M'))
    print('params=', str(params / 1000000.0) + '{}'.format('M'))


    output_FCN = output_FCN.cpu().detach().numpy().reshape(args.base_size, args.base_size)
    output_UNet = output_UNet.cpu().detach().numpy().reshape(args.base_size, args.base_size)
    output_Fusionnet = output_Fusionnet.cpu().detach().numpy().reshape(args.base_size, args.base_size)
    output_DeepLabv3_plus = output_DeepLabv3_plus.cpu().detach().numpy().reshape(args.base_size, args.base_size)
    output_ENet = output_ENet.cpu().detach().numpy().reshape(args.base_size, args.base_size)
    output_BiSeNet = output_BiSeNet.cpu().detach().numpy().reshape(args.base_size, args.base_size)
    output_DFA = output_DFA.cpu().detach().numpy().reshape(args.base_size, args.base_size)
    output_MDvsFA = output_MDvsFA.cpu().detach().numpy().reshape(args.base_size, args.base_size)
    output_ACM = output_ACM.cpu().detach().numpy().reshape(args.base_size, args.base_size)
    output_LSPM = output_LSPM.cpu().detach().numpy().reshape(args.base_size, args.base_size)
    output_DNA = output_DNA.cpu().detach().numpy().reshape(args.base_size, args.base_size)
    output_AGPC = output_AGPC.cpu().detach().numpy().reshape(args.base_size, args.base_size)
    output_LW_IRST = output_LW_IRST.cpu().detach().numpy().reshape(args.base_size, args.base_size)


    figure =plt.figure()
    plt.subplot(2,7,1), plt.imshow(img,cmap='gray'), plt.axis('off'), plt.title('Original Image ',fontsize=10)
    plt.subplot(2,7,2), plt.imshow(output_LW_IRST > 0, cmap='gray'), plt.axis('off'),  plt.title('LW_IRST',fontsize=10)
    plt.subplot(2,7,3), plt.imshow(output_FCN > 0, cmap='gray'), plt.axis('off'), plt.title('FCN',fontsize=10)
    plt.subplot(2,7,4), plt.imshow(output_UNet > 0, cmap='gray'), plt.axis('off'), plt.title('UNet',fontsize=10)
    plt.subplot(2,7,5), plt.imshow(output_Fusionnet > 0, cmap='gray'), plt.axis('off'), plt.title('Fusionnet',fontsize=10)
    plt.subplot(2,7,6), plt.imshow(output_DeepLabv3_plus > 0, cmap='gray'), plt.axis('off'), plt.title('DeepLabv3_plus',fontsize=10)
    plt.subplot(2,7,7), plt.imshow(output_ENet > 0, cmap='gray'), plt.axis('off'), plt.title('ENet',fontsize=10)
    plt.subplot(2,7,8), plt.imshow(output_BiSeNet > 0, cmap='gray'), plt.axis('off'), plt.title('BiSeNet',fontsize=10)
    plt.subplot(2,7,9), plt.imshow(output_DFA > 0, cmap='gray'), plt.axis('off'), plt.title('DFA',fontsize=10)
    plt.subplot(2,7,10), plt.imshow(output_MDvsFA > 0, cmap='gray'), plt.axis('off'), plt.title('MDvsFA',fontsize=10)
    plt.subplot(2,7,11), plt.imshow(output_ACM > 0, cmap='gray'), plt.axis('off'), plt.title('ACM',fontsize=10)
    plt.subplot(2,7,12), plt.imshow(output_LSPM > 0, cmap='gray'), plt.axis('off'), plt.title('LSPM',fontsize=10)
    plt.subplot(2,7,13), plt.imshow(output_DNA > 0, cmap='gray'), plt.axis('off'), plt.title('DNA',fontsize=10)
    plt.subplot(2,7,14), plt.imshow(output_AGPC > 0, cmap='gray'), plt.axis('off'), plt.title('AGPC',fontsize=10)
    plt.subplots_adjust(left=0.008,bottom=0.195,right=0.992,top=0.745,wspace=0.076,hspace=0.059)
    plt.show()

