
# coding: utf-8

# In[22]:
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

import imageio as io

import matplotlib.pyplot as plt


# In[23]:


class Encoder(nn.Module):
    def __init__(self,img_channels, latent_size):
        super(Encoder,self).__init__()
        self.img_channels = img_channels
        self.latent_size = latent_size
        self.conv1 = nn.Conv2d(in_channels = img_channels, out_channels = 32, kernel_size = 4, stride = 2)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 4, stride = 2)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 4, stride = 2)
        self.conv4 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 4, stride = 2)
        
        self.mu = nn.Linear(in_features = 2*2*256, out_features = latent_size)
        self.logsigma = nn.Linear(in_features = 2*2*256, out_features = latent_size)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        
        mu = self.mu(x)
        logsigma = self.logsigma(x)
        
        return mu, logsigma


# In[24]:


class Decoder(nn.Module):
    def __init__(self,img_channels, latent_size):
        super(Decoder,self).__init__()
        self.img_channels = img_channels
        self.latent_size = latent_size
        
        self.linear1 = nn.Linear(latent_size,1024)
        self.deconv1 = nn.ConvTranspose2d(in_channels = 1024, out_channels = 128, kernel_size = 5, stride = 2)
        self.deconv2 = nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = 5, stride = 2)
        self.deconv3 = nn.ConvTranspose2d(in_channels = 64, out_channels = 32, kernel_size = 6, stride = 2)
        self.deconv4 = nn.ConvTranspose2d(in_channels = 32, out_channels = img_channels, kernel_size = 6, stride = 2)
        
    def forward(self,z):
        z = F.relu(self.linear1(z))
        z = z.unsqueeze(-1).unsqueeze(-1)
        z = F.relu(self.deconv1(z))
        z = F.relu(self.deconv2(z))
        z = F.relu(self.deconv3(z))
        z = torch.sigmoid(self.deconv4(z))
        return z


# In[25]:


class VAE(nn.Module):
    def __init__(self,img_channels, latent_size):
        super(VAE,self).__init__()
        self.latent_size = latent_size
        self.encoder = Encoder(img_channels, latent_size)
        self.decoder = Decoder(img_channels, latent_size)
        
    def forward(self,x):
        mu,logsigma = self.encoder(x)
        
        sigma = logsigma.exp()
        epsilon = torch.randn_like(sigma)
        z = epsilon.mul(sigma).add_(mu)   
        
        recon_x = self.decoder(z)
        return recon_x, mu, logsigma


# $KL_{loss}=-\frac{1}{2}(2\log(\sigma_1)-\sigma_1^2-\mu_1^2+1)$  if σ is the standard deviation.   
# Warning, if σ if the variance, $=-\frac{1}{2}(\log(\sigma_1)-\sigma_1-\mu^2_1+1)$

# In[26]:


class ConvVAE():
    def __init__(self,img_channels, latent_size, learning_rate, load = False):
        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda else "cpu")
        print("ConVAE running on GPU" if self.cuda else "ConVAE running on CPU")
        self.model = VAE(img_channels, latent_size).to(self.device)
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.model.parameters(), lr = learning_rate)
        self.losses = []
        self.BCEs = []
        self.KLDs = []
        self.latent_size = latent_size
        self.epoch_trained = 0
        if load:
            self.load('./models/ConvVAE'+str(latent_size))
        
    def train(self, batch_img, batch_size, nb_epochs):
        demo = []
        demo.append(self.save_figure(batch_img))
        if self.epoch_trained>0: print(f'ConvVAE already trained on {self.epoch_trained} epochs')
        batch_img = batch_img.to(self.device)        
        for epoch in range(1,nb_epochs+1):
            loss_epoch = 0
            for batch in torch.split(batch_img, 40, dim=0):
                self.model.train()
                self.optimizer.zero_grad()
                recon_x, mu, logsigma = self.model(batch)
                
                BCE = F.mse_loss(recon_x, batch, reduction='sum')
                BCE_blue = F.mse_loss(recon_x[:,0,:,:], batch[:,0,:,:], reduction='sum')
                BCE_end = F.mse_loss(torch.split(recon_x,int(recon_x.size(2)/3),dim=2)[0], torch.split(batch,int(batch.size(2)/3),dim=2)[0], reduction='sum')
                #If the training is bad, add a threshold to KLD
                KLD = -0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())
                loss = BCE + KLD + BCE_end + BCE_blue
                loss_epoch+=loss
                self.losses.append(loss)
                self.BCEs.append(BCE)
                self.KLDs.append(KLD)
                    
                loss.backward()
                self.optimizer.step()
            
            if epoch%max(int(nb_epochs/10),1)==0:
                print(f'Epoch {epoch+self.epoch_trained}: loss = {round(float(loss_epoch),4)}') #CHECK THIS LINE, voir s on peut utiliser tdqm
                demo.append(self.save_figure(batch_img))
        self.epoch_trained += nb_epochs
        io.mimsave('./figures/training_epochs='+str(self.epoch_trained)+'_images='+str(batch_img.size(0))+'_latent='+str(self.latent_size)+'.gif', demo, duration = 0.55)            
        return recon_x
        
    def __call__(self,batch_img):
        return self.model(batch_img.to(self.device))[0]
    
    def encode(self,batch_img):
        mu, logsigma = self.model.encoder(batch_img.to(self.device))
        sigma = logsigma.exp()
        epsilon = torch.randn_like(sigma)
        z = epsilon.mul(sigma).add_(mu).detach()
        return z

    def decode(self,z):
        batch_img = self.model.decoder(z).detach()
        return batch_img

    def display_reconstruction(self,batch_img, id_img):
        recon_batch_img = self.model(batch_img[id_img,:,:,:].unsqueeze(0).to(self.device))[0].detach()
        import matplotlib.pyplot as plt
        
        plt.figure(figsize = (10,20))
        for channel in range(3):
            img = np.transpose(batch_img[id_img,:,:,:].numpy(),(1,2,0))
            plt.subplot(4,2,2*channel+1)
            plt.imshow(img[:,:,channel]*255)
            plt.axis('off')
            img = np.transpose(np.array(recon_batch_img[0,:,:,:].detach()*255),(1,2,0))
            plt.subplot(4,2,2*(channel+1))
            plt.imshow(np.clip(img,0,255)[:,:,channel])
            plt.axis('off')
            
        plt.subplot(4,2,7)
        plt.imshow(np.transpose(batch_img[id_img,:,:,:].numpy(),(1,2,0)))
        plt.axis('off')
        img = np.transpose(np.array(recon_batch_img[0,:,:,:].detach()),(1,2,0))
        plt.subplot(4,2,8)
        plt.imshow(np.clip(img,0,1))
        plt.axis('off')
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()
        
    def save_figure(self,batch_img):
        recon_batch_img = self.model(batch_img[0,:,:,:].unsqueeze(0).to(self.device))[0].detach()
        
        fig = plt.figure(figsize = (10,16))            
        plt.subplot(3,2,1)
        plt.imshow(np.transpose(batch_img[0,:,:,:].cpu().numpy(),(1,2,0)))
        plt.axis('off')
        img = np.transpose(np.array(recon_batch_img[0,:,:,:].detach().cpu()),(1,2,0))
        plt.subplot(3,2,2)
        plt.imshow(np.clip(img,0,1))
        plt.axis('off')
        
        recon_batch_img = self.model(batch_img[batch_img.size(0)//2,:,:,:].unsqueeze(0).to(self.device))[0].detach()
        plt.subplot(3,2,3)
        plt.imshow(np.transpose(batch_img[batch_img.size(0)//2,:,:,:].cpu().numpy(),(1,2,0)))
        plt.axis('off')
        img = np.transpose(np.array(recon_batch_img[0,:,:,:].detach().cpu()),(1,2,0))
        plt.subplot(3,2,4)
        plt.imshow(np.clip(img,0,1))
        plt.axis('off')
        
        recon_batch_img = self.model(batch_img[-1,:,:,:].unsqueeze(0).to(self.device))[0].detach()
        plt.subplot(3,2,5)
        plt.imshow(np.transpose(batch_img[-1,:,:,:].cpu().numpy(),(1,2,0)))
        plt.axis('off')
        img = np.transpose(np.array(recon_batch_img[0,:,:,:].detach().cpu()),(1,2,0))
        plt.subplot(3,2,6)
        plt.imshow(np.clip(img,0,1))
        plt.axis('off')
        
        plt.subplots_adjust(wspace=0, hspace=0)
        fig.tight_layout(pad=2)
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return data
        
        
    def save(self,path=None):
        path = './models/ConvVAE' if path==None else path
        torch.save(self.optimizer.state_dict(),path+'_optimizer.pt')
        torch.save(self.model.state_dict(),path+'_weights.pt')
        print('Model and Optimizer saved')
        
    def load(self,path=None):
        path = './models/ConvVAE' if path==None else path
        self.model.load_state_dict(torch.load(path+'_weights.pt', map_location=self.device))
        self.optimizer.load_state_dict(torch.load(path+'_optimizer.pt', map_location=self.device))
        self.model.eval()
        print('Model and Optimizer loaded')
        
    def plot_encoded(self, batch_z, encoded=True):
        batch_img = self.decode(batch_z) if encoded else batch_z
        for img in batch_img:
            plt.imshow(np.transpose(img, (1,2,0)))
            plt.axis('off')
            plt.show()