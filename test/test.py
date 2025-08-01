from net import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, DetailFeatureExtraction, FeatureAllocationFusion
import os
import numpy as np
from utils.Evaluator import Evaluator
import torch
import torch.nn as nn
from utils.img_read_save import img_save,image_read_cv2
import warnings
import logging
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ckpt_path=r"models/.pth"
for dataset_name in ["",""]:
    print("\n"*2+"="*80)
    model_name="    "
    print("The test result of "+dataset_name+' :')
    test_folder=os.path.join('',dataset_name) 
    test_out_folder=os.path.join('',dataset_name)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Encoder = nn.DataParallel(Restormer_Encoder()).to(device)
    Decoder = nn.DataParallel(Restormer_Decoder()).to(device)
    BaseFuseLayer = nn.DataParallel(BaseFeatureExtraction(dim=64, num_heads=8)).to(device)
    DetailFuseLayer = nn.DataParallel(DetailFeatureExtraction(num_layers=1)).to(device)
    FAF_Module = nn.DataParallel(FeatureAllocationFusion(dim=64)).to(device)

    Encoder.load_state_dict(torch.load(ckpt_path)['DIDF_Encoder'])
    Decoder.load_state_dict(torch.load(ckpt_path)['DIDF_Decoder'])
    BaseFuseLayer.load_state_dict(torch.load(ckpt_path)['BaseFuseLayer'])
    DetailFuseLayer.load_state_dict(torch.load(ckpt_path)['DetailFuseLayer'])
    FAF_Module.load_state_dict(torch.load(ckpt_path)['FAF_Module'])
    Encoder.eval()
    Decoder.eval()
    BaseFuseLayer.eval()
    DetailFuseLayer.eval()
    FAF_Module.eval()

    with torch.no_grad():
        for img_name in os.listdir(os.path.join(test_folder,"dth")):

            data_d=image_read_cv2(os.path.join(test_folder,"dth",img_name),mode='GRAY')[np.newaxis,np.newaxis, ...]/255.0
            data_i = image_read_cv2(os.path.join(test_folder,"sth",img_name), mode='GRAY')[np.newaxis,np.newaxis, ...]/255.0

            data_d,data_i = torch.FloatTensor(data_d),torch.FloatTensor(data_i)
            data_i, data_d = data_i.cuda(), data_d.cuda()

            F_I_L, F_I_H, feature_V = Encoder(data_i)
            F_D_L, F_D_H, feature_I = Encoder(data_d)
            
            F_L_fused, F_H_fused = FAF_Module(F_D_L, F_D_H, F_I_L, F_I_H)
            feature_F_B = BaseFuseLayer(F_L_fused)
            feature_F_D = DetailFuseLayer(F_H_fused)
            data_Fuse, _ = Decoder(data_i, feature_F_B, feature_F_D)
            data_Fuse=(data_Fuse-torch.min(data_Fuse))/(torch.max(data_Fuse)-torch.min(data_Fuse))
            fi = np.squeeze((data_Fuse * 255).cpu().numpy())
            img_save(fi, img_name.split(sep='.')[0], test_out_folder)


    eval_folder=test_out_folder  
    ori_img_folder=test_folder

    metric_result = np.zeros((8))
    for img_name in os.listdir(os.path.join(ori_img_folder,"dth")):
            ir = image_read_cv2(os.path.join(ori_img_folder,"dth", img_name), 'GRAY')
            vi = image_read_cv2(os.path.join(ori_img_folder,"sth", img_name), 'GRAY')
            fi = image_read_cv2(os.path.join(eval_folder, img_name.split('.')[0]+".png"), 'GRAY')
            metric_result += np.array([Evaluator.EN(fi), Evaluator.SD(fi)
                                        , Evaluator.SF(fi), Evaluator.MI(fi, ir, vi)
                                        , Evaluator.SCD(fi, ir, vi), Evaluator.VIFF(fi, ir, vi)
                                        , Evaluator.Qabf(fi, ir, vi), Evaluator.SSIM(fi, ir, vi)])

    metric_result /= len(os.listdir(eval_folder))
    print("\t\t EN\t SD\t SF\t MI\tSCD\tVIF\tQabf\tSSIM")
    print(model_name+'\t'+str(np.round(metric_result[0], 2))+'\t'
            +str(np.round(metric_result[1], 2))+'\t'
            +str(np.round(metric_result[2], 2))+'\t'
            +str(np.round(metric_result[3], 2))+'\t'
            +str(np.round(metric_result[4], 2))+'\t'
            +str(np.round(metric_result[5], 2))+'\t'
            +str(np.round(metric_result[6], 2))+'\t'
            +str(np.round(metric_result[7], 2))
            )
    print("="*80)