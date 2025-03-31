import os
import argparse
import datetime
import time
import importlib
import scipy.io as sio
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from os import makedirs
from os.path import join, exists
from scipy.io import loadmat,savemat
from models import network
# from core import train, train_cs, test
from core import train, test
from core.train import train_cs
from core.full_test import full_test
from generate_pic import aa_and_each_accuracy, sampling1, sampling2, sampling3, load_dataset, generate_png, \
    generate_iter, classification_map
from generate_pic import generate_train_iter, generate_valida_iter, generate_test_iter, generate_all_iter, generate_full_iter, generate_iter,generate_train_known_iter,generate_test_known_iter,generate_test_unknown_iter,generate_fulltest_iter
import numpy as np
from sklearn import metrics, preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import cohen_kappa_score

parser = argparse.ArgumentParser("Training")

# Dataset
parser.add_argument('--dataset', choices=['SA', 'PU', 'IP'], default='SA', help='dataset to use')
parser.add_argument('--patches', type=int, default=7, help='number of patches')#todo 5
parser.add_argument('--num', type=int, default=30, help='number of samples')


# optimization
parser.add_argument('--batch-size', type=int, default=1)#todo 200->20
parser.add_argument('--lr', type=float, default=0.001, help="learning rate for model")
parser.add_argument('--experiment_num', type=int, default=10)#todo 10->
parser.add_argument('--max-epoch', type=int, default=100)#todo 100->
parser.add_argument('--stepsize', type=int, default=200)
parser.add_argument('--temp', type=float, default=1.0, help="temp")
parser.add_argument('--num-centers', type=int, default=1)

# model
parser.add_argument('--weight-pl', type=float, default=0.1, help="weight for center loss")
parser.add_argument('--beta', type=float, default=0.1, help="weight for entropy loss")
parser.add_argument('--model', type=str, default='SSMLP-RPL')
parser.add_argument('--layers', type=int, default=4)
parser.add_argument('--embed_dims', type=int, default=64)
parser.add_argument('--segment_dim', type=int, default=8)


# misc
parser.add_argument('--nz', type=int, default=100)
parser.add_argument('--ns', type=int, default=1)
parser.add_argument('--eval-freq', type=int, default=100)
parser.add_argument('--print-freq', type=int, default=100)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--save-dir', type=str, default='../log')
parser.add_argument('--loss', type=str, default='ARPLoss')
parser.add_argument('--eval', action='store_true', help="Eval", default=False)
parser.add_argument('--cs', action='store_true', help="Confusing Sample", default=False)


def get_accuracy(y_true, y_pred):
    num_perclass = np.zeros(int(y_true.max() + 1))
    num = np.zeros(int(y_true.max() + 1))
    for i in range(len(y_true)):
        num_perclass[int(y_true[i])] += 1#todo 统计在GT里面的每一类的标签有多少个
    for i in range(len(y_pred)):
        if y_pred[i] == y_true[i]:#todo 预测和GT对应的标签是的一样的，说明预测正确，
            num[int(y_pred[i])] += 1#todo 统计预测正确的类别的个数（pred和true的位置是对应的）
    OA = num.sum()/num_perclass.sum()#add todo 预测正确的类别个数除以所有的标签OA = 样本中正确分类的总数除以样本总数
    for i in range(len(num)):
        if(num_perclass[i]==0):#add todo 防止被除数为0
            num[i] = 0
            continue
        num[i] = num[i] / num_perclass[i]#todo 每一类的分类正确的精度
    # for i in range(len(num)):
    #     print('class' + i +' : '+num[i])
    AA = num.sum()/len(num)#todo AA = 每一类别中预测正确的数目除以该类总数，记为该类的精度，最后求每类精度的平均
    X = len(num)

    acc = accuracy_score(y_true, y_pred)#todo acc = OA
    kappa = cohen_kappa_score(y_true, y_pred)

    ac = np.zeros(int(y_true.max() + 1 + 2))
    ac[:int(y_true.max() + 1)] = num#todo 将ac的（0-15）用num填充（0-15类的每一类的精度），16填放OA，17填放Kappa
    ac[-1] = acc
    ac[-2] = kappa
    return ac  # acc,num.mean(),kappa

if __name__ == '__main__':
    args = parser.parse_args()
    options = vars(args)
    results = dict()


    hsi_path = 'data/salinas/salinas_corrected.mat'
    # gt_path = 'data/salinas/salinas_gt.mat'
    gt_path = 'data/salinas/salinas17.mat'


    dataname=args.dataset
    model_name='SSMLP-RPL'

    data_hsi=sio.loadmat(hsi_path)
    data_hsi=data_hsi['salinas_corrected']
    gt=sio.loadmat(gt_path)
    # gt=gt['salinas_gt']#todo 512X217
    gt=gt['array']#todo 512X217


    # training samples per class
    SAMPLES_NUM = args.num#todo 30
    experiment_num=args.experiment_num#todo 10
    ROWS, COLUMNS, BAND = data_hsi.shape#todo 512X217X204
    data = data_hsi.reshape(np.prod(data_hsi.shape[:2]), np.prod(data_hsi.shape[2:]))#todo 111104X204(512X217 = 111104)
    gt2 = gt.reshape(np.prod(gt.shape[:2]), )#todo 111104(将gt展平)
    CLASSES_NUM = gt.max()#todo 16->9
    print('The class numbers of the HSI data is:', CLASSES_NUM)

    # known = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]  # define the known classes
    # print('The known class of the HSI data is:', known)
    # unknown = list(set(list(range(0,  gt.max()))) - set(known))
    # print('The unknown class of the HSI data is:', unknown)
    known = [0, 1, 2, 3, 4, 5, 6, 7, 8,9,10,11,12,13,14,15]  # define the known classes
    print('The known class of the HSI data is:', known)
    unknown = list(set(list(range(0,  gt.max()))) - set(known))
    print('The unknown class of the HSI data is:', unknown)

    print('-----Importing Setting Parameters-----')

    patch_list=[7]

    for i_patch in range(len(patch_list)):
        patches=patch_list[i_patch]#todo 7
        PATCH_LENGTH = int((patches-1)/2)#todo 3
        # number of training samples per class
        # lr, num_epochs, batch_size = 0.0001, 200, 32


        img_rows = 2 * PATCH_LENGTH + 1#todo 7
        img_cols = 2 * PATCH_LENGTH + 1#todo 7
        INPUT_DIMENSION = data_hsi.shape[2]#todo 200
        FULL_SIZE = data_hsi.shape[0] * data_hsi.shape[1]#todo 21025
        ALL_SIZE = data_hsi.shape[0] * data_hsi.shape[1]#todo 21025

        data = preprocessing.scale(data)#todo (21025,200)
        whole_data = data.reshape(data_hsi.shape[0], data_hsi.shape[1], data_hsi.shape[2])#todo (145,145,200)

        padded_data = np.lib.pad(whole_data, ((PATCH_LENGTH, PATCH_LENGTH), (PATCH_LENGTH, PATCH_LENGTH), (0, 0)),'symmetric')#todo (151,151,200)

        Experiment_result = np.zeros([CLASSES_NUM + 7, experiment_num + 2])#todo (23,3)
        for iter_num in range(experiment_num):
            np.random.seed(iter_num+123456)#todo seed
            train_indices, test_indices = sampling1(SAMPLES_NUM, gt2, options['dataset'])#todo 采样
            #todo train_indices list:480, test_indices list:52649
            full_indices = sampling3(gt2, 1)#todo list:111104
            TRAIN_SIZE = len(train_indices)#todo 480
            VAL_SIZE = int(TRAIN_SIZE)#todo 480
            TEST_SIZE = len(test_indices)#todo 53649
            if  unknown == None:
                #full_iter = generate_full_iter(whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION, args.batch_size, gt2, FULL_SIZE,full_indices)
                train_iter = generate_train_iter(TRAIN_SIZE, train_indices, whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION,args.batch_size, gt2)
                test_iter = generate_test_iter(TEST_SIZE, test_indices, VAL_SIZE, whole_data, PATCH_LENGTH, padded_data,INPUT_DIMENSION, args.batch_size, gt2)
                known_test_iter = None
                unknown_test_iter = None
            else:
                full_iter = generate_full_iter(whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION, args.batch_size, gt2,FULL_SIZE, full_indices)
                test_iter = generate_fulltest_iter(TEST_SIZE, test_indices, VAL_SIZE, whole_data, PATCH_LENGTH, padded_data,INPUT_DIMENSION, args.batch_size, gt2, known, CLASSES_NUM-1)
                train_iter = generate_train_known_iter(TRAIN_SIZE, train_indices, whole_data, PATCH_LENGTH, padded_data,INPUT_DIMENSION, args.batch_size, gt2, known, augmentation=True)
                known_test_iter = generate_test_known_iter(TEST_SIZE, test_indices, VAL_SIZE, whole_data, PATCH_LENGTH, padded_data,INPUT_DIMENSION, args.batch_size, gt2, known)
                unknown_test_iter = generate_test_unknown_iter(TEST_SIZE, test_indices, VAL_SIZE, whole_data, PATCH_LENGTH, padded_data,INPUT_DIMENSION, args.batch_size, gt2, unknown, CLASSES_NUM-1)

            options.update(
                {
                    'BAND':BAND,
                    'known': known,
                    'unknown': unknown,
                    'full_iter' : full_iter,
                    'test_iter' : test_iter,
                    'train_iter': train_iter,
                    'known_test_iter' : known_test_iter,
                    'unknown_test_iter' : unknown_test_iter,

            }
            )

            torch.manual_seed(options['seed'])
            os.environ['CUDA_VISIBLE_DEVICES'] = options['gpu']
            use_gpu = torch.cuda.is_available()
            if options['use_cpu']: use_gpu = False

            if use_gpu:
                print("Currently using GPU: {}".format(options['gpu']))
                cudnn.benchmark = True
                torch.cuda.manual_seed_all(options['seed'])
            else:
                print("Currently using CPU")

            # Dataset
            print("{} Preparation".format(options['dataset']))
            if options['unknown'] == None:
                trainloader, testloader, outloader = options['train_iter'], options['test_iter'], None
            else:
                trainloader, testloader, full_loader, known_test_loader, unknown_test_loader = options['train_iter'], options['test_iter'], \
                                                                                   options['full_iter'], options[
                                                                                       'known_test_iter'], options[
                                                                                       'unknown_test_iter']

            # unknow=options['unknown'][0]
            # Model
            print("Creating model: {}".format(options['model']))

            options['num_classes'] = len(options['known'])
            unknow = options['num_classes']

            net = network.SSMLP(patches, options['BAND'], options['num_classes'], layers=options['layers'], embed_dims=options['embed_dims'],segment_dim=options['segment_dim'])
            feat_dim = options['embed_dims']
            # feat_dim = 128

            # Loss
            options.update(
                {
                    'feat_dim': feat_dim,
                    'use_gpu': use_gpu
                }
            )

            Loss = importlib.import_module('loss.' + options['loss'])
            criterion = getattr(Loss, options['loss'])(**options)

            if use_gpu:
                net = nn.DataParallel(net).cuda()
                criterion = criterion.cuda()

            params_list = [{'params': net.parameters()},
                           {'params': criterion.parameters()}]

            # optimizer = torch.optim.SGD(params_list, lr=options['lr'], momentum=0.9, weight_decay=1e-4)
            optimizer = torch.optim.Adam(params_list, lr=options['lr'])#todo 优化器

            if options['stepsize'] > 0:
                scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90, 120])

            start_time = time.time()

            for epoch in range(options['max_epoch']):
                print("==> Epoch {}/{}".format(epoch + 1, options['max_epoch']))

                _, logits_min, dis_min, loss_r = train.train(net, criterion, optimizer, trainloader, epoch=epoch, **options)
#todo train()->train.train()
                #if options['eval_freq'] > 0 and (epoch + 1) % options['eval_freq'] == 0 or (epoch + 1) == options['max_epoch']:
                #    print("==> Test", options['loss'])
                    #results, pred, label = test(net, criterion, full_testloader, testloader, outloader, logits_min, dis_min,
                    #                            loss_r, unknow, epoch=epoch, **options)
                    #print("Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(results['ACC'], results['AUROC'],
                    #                                                                        results['OSCR']))

                    # save_networks(net, model_path, file_name, criterion=criterion)
                if options['stepsize'] > 0: scheduler.step()

            train_time2=time.time()
            tes_time1=time.time()
            results, pred, label = test.test(net, criterion, testloader, known_test_loader, unknown_test_loader, logits_min, dis_min,loss_r, unknow, epoch=epoch, **options)
#todo test()->test.test()
            results_test, pred_test, label_test = test.test(net, criterion, full_loader, known_test_loader, unknown_test_loader, logits_min, dis_min,loss_r, unknow, epoch=epoch, **options)

            print("Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(results['ACC'], results['AUROC'],results['OSCR']))
            tes_time2=time.time()

            #pred_global = full_test(net, criterion, full_testloader, full_loader, testloader, outloader, logits_min, dis_min,loss_r, unknow, epoch=epoch, **options)
            #generate_png(gt, pred_global, dataname, ROWS, COLUMNS, SAMPLES_NUM)
            # todo add 保存结果
            # result_save_path = "result/{}/{}/{}/".format(args.dataset, args.num, args.patches)
            # if not os.path.isdir(result_save_path):
            #     os.makedirs(result_save_path)
            # savemat(os.path.join(result_save_path, 'result_{}.mat'.format(args.run_number)),
            #         {'pred': pre_gsrl, 'gt': all_gt})
            # print(result_save_path)
            # add todo

            # #todo label 9781 (0-15) , pred 9781 (0-14)
            ac = get_accuracy(label, pred)
            label_1 = label
            pred_1 = pred
            # num_perclass = np.zeros(int(label.max() + 1))
            # num = np.zeros(int(label.max() + 1))
            # for i in range(len(label)):
            #     num_perclass[int(label[i])] += 1  # todo 统计在GT里面的每一类的标签有多少个
            # for i in range(len(pred)):
            #     if y_pred[i] == y_true[i]:  # todo 预测和GT对应的标签是的一样的，说明预测正确，
            #         num[int(y_pred[i])] += 1  # todo 统计预测正确的类别的个数（pred和true的位置是对应的）
            # OA = num.sum() / num_perclass.sum()  # add todo 预测正确的类别个数除以所有的标签OA = 样本中正确分类的总数除以样本总数
            # for i in range(len(num)):
            #     if (num_perclass[i] == 0):  # add todo 防止被除数为0
            #         num[i] = 0
            #         continue
            #     num[i] = num[i] / num_perclass[i]  # todo 每一类的分类正确的精度
            # for i in range(len(num)):
            #     print('class' + i + ' : ' + num[i])
            # AA = num.sum() / len(num)  # todo AA = 每一类别中预测正确的数目除以该类总数，记为该类的精度，最后求每类精度的平均
            # X = len(num)

            # todo confusion_matrix
            print(confusion_matrix(label, pred))


            elapsed = round(time.time() - start_time)
            elapsed = str(datetime.timedelta(seconds=elapsed))
            print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

            Experiment_result[0, iter_num] = ac[-1] * 100  # OA
            Experiment_result[1, iter_num] = np.mean(ac[:-2]) * 100  # AA
            Experiment_result[2, iter_num] = ac[-2] * 100  # Kappa
            Experiment_result[3, iter_num] = results['ACC']                           #Closed-set  OA
            Experiment_result[4, iter_num] = results['OSCR']                          #OSCR
            Experiment_result[5, iter_num] = train_time2 - start_time
            Experiment_result[6, iter_num] = tes_time2 - tes_time1
            Experiment_result[7:, iter_num] = ac[:-2] * 100#todo 每一类的精度

            print('########### Experiment {}，Model assessment Finished！ ###########'.format(iter_num))

            ########## mean value & standard deviation #############

        Experiment_result[:, -2] = np.mean(Experiment_result[:, 0:-2], axis=1)  # 计算均值
        Experiment_result[:, -1] = np.std(Experiment_result[:, 0:-2], axis=1)  # 计算平均差

        pred_mat = pred_test.reshape(gt.shape[0], gt.shape[1])
        save_dir = 'new_mask'
        if not exists(save_dir):
            makedirs(save_dir)
        # sio.savemat(join(save_dir, 'a.mat'), {'y_pred_first': y_pred_first})
        sio.savemat(join(save_dir, 'ssmlp_SA.mat'), {'mask': pred_mat})

        day = datetime.datetime.now()
        day_str = day.strftime('%m_%d_%H_%M')
        #add todo
        for i in range(0,gt.max()):
            print(f'第 {i} 类:{Experiment_result[i+7,-2]}')
        print('OA : ' + str((Experiment_result[0,-2])))
        print('AA : ' + str((Experiment_result[1, -2])))
        print('Kappa : ' + str(Experiment_result[2, -2]))
        # generate_png(gt,pred,args.dataset,gt.shape[0],gt.shape[1],args.num)
        generate_png(gt,pred_test,args.dataset,gt.shape[0],gt.shape[1],args.num)
