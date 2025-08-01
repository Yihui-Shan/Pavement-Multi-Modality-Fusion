------------------------------------------------------------------------------
Train
------------------------------------------------------------------------------
'''

step = 0
torch.backends.cudnn.benchmark = True
prev_time = time.time()

for epoch in range(num_epochs):
    ''' train '''
    for i, (data_i, data_d) in enumerate(loader['train']):
        data_i, data_d = data_i.cuda(), data_d.cuda()
        DIDF_Encoder.train()
        DIDF_Decoder.train()
        BaseFuseLayer.train()
        DetailFuseLayer.train()
        FAF_Module.train()

        DIDF_Encoder.zero_grad()
        DIDF_Decoder.zero_grad()
        BaseFuseLayer.zero_grad()
        DetailFuseLayer.zero_grad()
        FAF_Module.zero_grad()

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        optimizer4.zero_grad()
        optimizer5.zero_grad()

        if epoch < epoch_gap: #Phase I
            F_I_L, F_I_H, _ = DIDF_Encoder(data_i)
            F_D_L, F_D_H, _ = DIDF_Encoder(data_d)
            data_i_hat, _ = DIDF_Decoder(data_i, F_I_L, F_I_H)
            data_d_hat, _ = DIDF_Decoder(data_d, F_D_L, F_D_H)

            # 使用StageOneLoss: L_one = L_mse + α1 * L_ssim + α2 * L_fd
            loss_one, mse_loss, ssim_loss, fd_loss = stage_one_loss(
                data_d, data_i, data_d_hat, data_i_hat, F_H_I=F_I_H, F_H_D=F_D_H, F_L_I=F_I_L, F_L_D=F_D_L
            )
            
            # 添加梯度损失以保持边缘信息
            gradient_loss = L1Loss(kornia.filters.SpatialGradient()(data_i),
                                   kornia.filters.SpatialGradient()(data_i_hat))
            
            # 总损失 = StageOneLoss + 梯度损失
            loss = loss_one + coeff_tv * gradient_loss

            loss.backward()
            nn.utils.clip_grad_norm_(
                DIDF_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                DIDF_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            optimizer1.step()  
            optimizer2.step()
        else:  #Phase II
            F_I_L, F_I_H, feature_V = DIDF_Encoder(data_i)
            F_D_L, F_D_H, feature_I = DIDF_Encoder(data_d)
            
            # 使用FAF模块进行特征分配融合
            F_L_fused, F_H_fused = FAF_Module(F_D_L, F_D_H, F_I_L, F_I_H)
            
            # 使用融合后的特征进行解码
            feature_F_B = BaseFuseLayer(F_L_fused)
            feature_F_D = DetailFuseLayer(F_H_fused)
            data_Fuse, feature_F = DIDF_Decoder(data_i, feature_F_B, feature_F_D)  

            # 使用StageTwoLoss: L_two = L_int + α3 * L_grad + α4 * L_fd
            loss_two, int_loss, grad_loss, fd_loss = stage_two_loss(
                data_Fuse, data_d, data_i, F_H_I=F_I_H, F_H_D=F_D_H, F_L_I=F_I_L, F_L_D=F_D_L
            )
            
            # 添加融合损失
            fusionloss, _,_  = criteria_fusion(data_i, data_d, data_Fuse)
            
            # 总损失 = StageTwoLoss + 融合损失
            loss = loss_two + fusionloss
            loss.backward()
            nn.utils.clip_grad_norm_(
                DIDF_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                DIDF_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                BaseFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                DetailFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                FAF_Module.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            optimizer1.step()  
            optimizer2.step()
            optimizer3.step()
            optimizer4.step()
            optimizer5.step()

        # Determine approximate time left
        batches_done = epoch * len(loader['train']) + i
        batches_left = num_epochs * len(loader['train']) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [loss: %f] ETA: %.10s"
            % (
                epoch,
                num_epochs,
                i,
                len(loader['train']),
                loss.item(),
                time_left,
            )
        )

    # adjust the learning rate

    scheduler1.step()  
    scheduler2.step()
    if not epoch < epoch_gap:
        scheduler3.step()
        scheduler4.step()
        scheduler5.step()

    if optimizer1.param_groups[0]['lr'] <= 1e-6:
        optimizer1.param_groups[0]['lr'] = 1e-6
    if optimizer2.param_groups[0]['lr'] <= 1e-6:
        optimizer2.param_groups[0]['lr'] = 1e-6
    if optimizer3.param_groups[0]['lr'] <= 1e-6:
        optimizer3.param_groups[0]['lr'] = 1e-6
    if optimizer4.param_groups[0]['lr'] <= 1e-6:
        optimizer4.param_groups[0]['lr'] = 1e-6
    if optimizer5.param_groups[0]['lr'] <= 1e-6:
        optimizer5.param_groups[0]['lr'] = 1e-6
    
if True:
    checkpoint = {
        'DIDF_Encoder': DIDF_Encoder.state_dict(),
        'DIDF_Decoder': DIDF_Decoder.state_dict(),
        'BaseFuseLayer': BaseFuseLayer.state_dict(),
        'DetailFuseLayer': DetailFuseLayer.state_dict(),
        'FAF_Module': FAF_Module.state_dict(),
    }
    torch.save(checkpoint, os.path.join("models/PMMF_"+timestamp+'.pth'))
