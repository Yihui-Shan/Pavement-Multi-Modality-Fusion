import torch
import torch.nn as nn
import torch.nn.functional as F



class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self,image_vis,image_ir,generate_img):
        image_y=image_vis[:,:1,:,:]
        x_in_max=torch.max(image_y,image_ir)
        loss_in=F.l1_loss(x_in_max,generate_img)
        y_grad=self.sobelconv(image_y)
        ir_grad=self.sobelconv(image_ir)
        generate_img_grad=self.sobelconv(generate_img)
        x_grad_joint=torch.max(y_grad,ir_grad)
        loss_grad=F.l1_loss(x_grad_joint,generate_img_grad)
        loss_total=loss_in+10*loss_grad
        return loss_total,loss_in,loss_grad

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)


def cc(img1, img2):
    eps = torch.finfo(torch.float32).eps
    """Correlation coefficient for (N, C, H, W) image; torch.float32 [0.,1.]."""
    N, C, _, _ = img1.shape
    img1 = img1.reshape(N, C, -1)
    img2 = img2.reshape(N, C, -1)
    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)
    cc = torch.sum(img1 * img2, dim=-1) / (eps + torch.sqrt(torch.sum(img1 **
                                                                      2, dim=-1)) * torch.sqrt(torch.sum(img2**2, dim=-1)))
    cc = torch.clamp(cc, -1., 1.)
    return cc.mean()


class StageOneLoss(nn.Module):
    """
    Stage-one loss function: L_one = L_mse + α1 * L_ssim + α2 * L_fd
    """
    def __init__(self, alpha1=1.0, alpha2=1.0, c1=0.01, c2=0.03, eps=1.01):
        super(StageOneLoss, self).__init__()
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.c1 = c1
        self.c2 = c2
        self.eps = eps
    
    def forward(self, D, I, D_hat, I_hat, F_H_I, F_H_D, F_L_I, F_L_D):
        """
        Args:
            D: input depth image
            I: input intensity image  
            D_hat: reconstructed depth image
            I_hat: reconstructed intensity image
            F_H_I: high-frequency features of intensity image
            F_H_D: high-frequency features of depth image
            F_L_I: low-frequency features of intensity image
            F_L_D: low-frequency features of depth image
        """
        # MSE Loss
        l_mse = self.mse_loss(D, I, D_hat, I_hat)
        
        # SSIM Loss
        l_ssim = self.ssim_loss(D, I, D_hat, I_hat)
        
        # Feature Decomposition Loss
        l_fd = self.feature_decomposition_loss(F_H_I, F_H_D, F_L_I, F_L_D)
        
        # Total loss
        l_one = l_mse + self.alpha1 * l_ssim + self.alpha2 * l_fd
        
        return l_one, l_mse, l_ssim, l_fd
    
    def mse_loss(self, D, I, D_hat, I_hat):
        """
        MSE Loss: L_mse = Σ((D_i - D_hat_i)^2 + (I_i - I_hat_i)^2)
        """
        mse_D = F.mse_loss(D, D_hat)
        mse_I = F.mse_loss(I, I_hat)
        return mse_D + mse_I
    
    def ssim_loss(self, D, I, D_hat, I_hat):
        """
        SSIM Loss: L_ssim = Σ(SSIM(D, D_hat) + SSIM(I, I_hat))
        """
        ssim_D = self.ssim(D, D_hat)
        ssim_I = self.ssim(I, I_hat)
        return ssim_D + ssim_I
    
    def ssim(self, x, y):
        """
        Calculate SSIM between two images
        SSIM = (2*μ_x*μ_y + c1)*(2*σ_xy + c2) / ((μ_x^2 + μ_y^2 + c1)*(σ_x^2 + σ_y^2 + c2))
        """
        # Calculate means
        mu_x = torch.mean(x, dim=[2, 3], keepdim=True)
        mu_y = torch.mean(y, dim=[2, 3], keepdim=True)
        
        # Calculate variances
        var_x = torch.var(x, dim=[2, 3], keepdim=True, unbiased=False)
        var_y = torch.var(y, dim=[2, 3], keepdim=True, unbiased=False)
        
        # Calculate covariance
        # σ_xy = E[(x-μ_x)(y-μ_y)] = E[xy] - μ_x*μ_y
        mu_xy = torch.mean(x * y, dim=[2, 3], keepdim=True)
        cov_xy = mu_xy - mu_x * mu_y
        
        # SSIM calculation
        numerator = (2 * mu_x * mu_y + self.c1) * (2 * cov_xy + self.c2)
        denominator = (mu_x**2 + mu_y**2 + self.c1) * (var_x + var_y + self.c2)
        
        ssim = numerator / (denominator + 1e-8)
        return torch.mean(ssim)
    
    def feature_decomposition_loss(self, F_H_I, F_H_D, F_L_I, F_L_D):
        """
        Feature Decomposition Loss: L_fd = (FCC(F_H_I, F_H_D))^2 / (FCC(F_L_I, F_L_D) + ε)
        where FCC is the correlation coefficient operator
        """
        # Calculate correlation coefficients
        fcc_high = self.fcc(F_H_I, F_H_D)
        fcc_low = self.fcc(F_L_I, F_L_D)
        
        # Feature decomposition loss
        l_fd = (fcc_high ** 2) / (fcc_low + self.eps)
        return l_fd
    
    def fcc(self, x, y):
        """
        Correlation coefficient operator FCC
        """
        # Flatten spatial dimensions
        x_flat = x.view(x.size(0), x.size(1), -1)  # (N, C, H*W)
        y_flat = y.view(y.size(0), y.size(1), -1)  # (N, C, H*W)
        
        # Calculate means
        x_mean = torch.mean(x_flat, dim=2, keepdim=True)
        y_mean = torch.mean(y_flat, dim=2, keepdim=True)
        
        # Center the data
        x_centered = x_flat - x_mean
        y_centered = y_flat - y_mean
        
        # Calculate correlation coefficient
        numerator = torch.sum(x_centered * y_centered, dim=2)
        denominator = torch.sqrt(torch.sum(x_centered ** 2, dim=2) * torch.sum(y_centered ** 2, dim=2))
        
        fcc = numerator / (denominator + 1e-8)
        fcc = torch.clamp(fcc, -1.0, 1.0)
        
        return torch.mean(fcc)


class StageTwoLoss(nn.Module):
    """
    Stage-two loss function: L_two = L_int + α3 * L_grad + α4 * L_fd
    """
    def __init__(self, alpha3=1.0, alpha4=1.0):
        super(StageTwoLoss, self).__init__()
        self.alpha3 = alpha3
        self.alpha4 = alpha4
        self.sobel_conv = Sobelxy()
    
    def forward(self, F, D, I, F_H_I, F_H_D, F_L_I, F_L_D):
        """
        Args:
            F: fused image
            D: depth image
            I: intensity image
            F_H_I: high-frequency features of intensity image
            F_H_D: high-frequency features of depth image
            F_L_I: low-frequency features of intensity image
            F_L_D: low-frequency features of depth image
        """
        # Intensity Loss
        l_int = self.intensity_loss(F, D, I)
        
        # Gradient Loss
        l_grad = self.gradient_loss(F, D, I)
        
        # Feature Decomposition Loss (reuse from StageOneLoss)
        l_fd = self.feature_decomposition_loss(F_H_I, F_H_D, F_L_I, F_L_D)
        
        # Total loss
        l_two = l_int + self.alpha3 * l_grad + self.alpha4 * l_fd
        
        return l_two, l_int, l_grad, l_fd
    
    def intensity_loss(self, F, D, I):
        """
        Intensity Loss: L_int = (1/N) * Σ|F_i - M_di*D_i - M_ii*I_i|
        where M_di and M_ii are depth and intensity masks respectively
        """
        # Calculate regional mean of depth image
        depth_mean = torch.mean(D)
        
        # Create depth mask: M_di = 1 if D_i >= mean, 0 otherwise
        M_di = (D >= depth_mean).float()
        
        # Create intensity mask: M_ii = 1 - M_di
        M_ii = 1.0 - M_di
        
        # Calculate intensity loss
        l_int = torch.mean(torch.abs(F - M_di * D - M_ii * I))
        
        return l_int
    
    def gradient_loss(self, F, D, I):
        """
        Gradient Loss: L_grad = (1/N) * Σ|∇F_i - M_di*∇D_i - M_ii*∇I_i|
        where ∇ indicates the Sobel gradient operator
        """
        # Calculate regional mean of depth image
        depth_mean = torch.mean(D)
        
        # Create depth mask: M_di = 1 if D_i >= mean, 0 otherwise
        M_di = (D >= depth_mean).float()
        
        # Create intensity mask: M_ii = 1 - M_di
        M_ii = 1.0 - M_di
        
        # Calculate Sobel gradients
        grad_F = self.sobel_conv(F)
        grad_D = self.sobel_conv(D)
        grad_I = self.sobel_conv(I)
        
        # Calculate gradient loss
        l_grad = torch.mean(torch.abs(grad_F - M_di * grad_D - M_ii * grad_I))
        
        return l_grad
    
    def feature_decomposition_loss(self, F_H_I, F_H_D, F_L_I, F_L_D):
        """
        Feature Decomposition Loss: L_fd = (FCC(F_H_I, F_H_D))^2 / (FCC(F_L_I, F_L_D) + ε)
        where FCC is the correlation coefficient operator
        """
        # Calculate correlation coefficients
        fcc_high = self.fcc(F_H_I, F_H_D)
        fcc_low = self.fcc(F_L_I, F_L_D)
        
        # Feature decomposition loss
        l_fd = (fcc_high ** 2) / (fcc_low + 1.01)  # ε = 1.01
        return l_fd
    
    def fcc(self, x, y):
        """
        Correlation coefficient operator FCC
        """
        # Flatten spatial dimensions
        x_flat = x.view(x.size(0), x.size(1), -1)  # (N, C, H*W)
        y_flat = y.view(y.size(0), y.size(1), -1)  # (N, C, H*W)
        
        # Calculate means
        x_mean = torch.mean(x_flat, dim=2, keepdim=True)
        y_mean = torch.mean(y_flat, dim=2, keepdim=True)
        
        # Center the data
        x_centered = x_flat - x_mean
        y_centered = y_flat - y_mean
        
        # Calculate correlation coefficient
        numerator = torch.sum(x_centered * y_centered, dim=2)
        denominator = torch.sqrt(torch.sum(x_centered ** 2, dim=2) * torch.sum(y_centered ** 2, dim=2))
        
        fcc = numerator / (denominator + 1e-8)
        fcc = torch.clamp(fcc, -1.0, 1.0)
        
        return torch.mean(fcc)