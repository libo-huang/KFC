from torch import nn
import torch

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def weights_init_conv(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Acc_Mnist(nn.Module):
    def __init__(self, args):
        super(Acc_Mnist, self).__init__()
        self.feat_dim = args.feat_dim
        self.latent_dim = args.latent_dim
        self.class_dim = args.class_dim
        self.model = nn.Sequential(
            nn.Linear(self.feat_dim, self.class_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.model(x)
        return y

class ConditionalVAE(nn.Module):
    def __init__(self, args):
        super(ConditionalVAE, self).__init__()
        self.feat_dim = args.feat_dim
        self.latent_dim = args.latent_dim
        self.hidden_dim = args.hidden_dim
        self.class_dim = args.class_dim

        self.model_encoder = nn.Sequential(
            nn.Linear(self.feat_dim+self.class_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
        )
        self.fc_mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.hidden_dim, self.latent_dim)
        self.model_decoder = nn.Sequential(
            nn.Linear(self.latent_dim+self.class_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.feat_dim),
        )
        self.sigmoid = nn.LeakyReLU()  # 因为Mnist像素值在0-1之间，所以建议用Sigmoid()

    def encode(self, x, y):
        x = torch.cat((x, y), 1)
        x = self.model_encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)
        return mu, logvar

    def decode(self, z, y):
        z = torch.cat((z, y), 1)
        logit = self.model_decoder(z)
        feat = self.sigmoid(logit)
        return feat

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        z = eps.mul(std).add_(mu)
        return z

    def forward(self, x, y):
        y = y.float()
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        x_rec = self.decode(z, y)
        return x_rec, mu, logvar

class ConditionalVAE_conv(nn.Module):
    def __init__(self, args):
        super(ConditionalVAE_conv, self).__init__()
        self.latent_dim = args.latent_dim  # TODO: 128
        self.class_dim = args.class_dim
        self.in_channel = 3 if args.dataset == 'cifar10' or args.dataset == 'svhn' else 1
        self.img_size = 32 # cifar10 3*32*32
        # self.embed_class = nn.Linear(args.class_dim, self.img_size * self.img_size * self.in_channel)
        self.embed_class = nn.Linear(args.class_dim, self.img_size * self.img_size)
        self.embed_data = nn.Conv2d(self.in_channel,self.in_channel,kernel_size=1)

        self.model_encoder = nn.Sequential(
            nn.Conv2d(self.in_channel + 1, 32, 3, 2, 1),
            # nn.Conv2d(self.in_channel + self.in_channel, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )

        self.fc_mu = nn.Linear(256*2*2, self.latent_dim)
        self.fc_var = nn.Linear(256*2*2, self.latent_dim)
        self.decoder_input = nn.Linear(self.latent_dim + args.class_dim, 256*2*2)

        self.model_decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU()
        )
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )
        #self.apply(weights_init_conv)

    def encode(self, x, y):
        y = self.embed_class(y)
        y = y.view(-1, self.img_size, self.img_size).unsqueeze(1)
        # y = y.view(-1, self.in_channel, self.img_size, self.img_size)
        x = self.embed_data(x)
        x = torch.cat([x, y], dim = 1)
        x = self.model_encoder(x)  # TODO
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)
        return mu, logvar

    def decode(self, z, y):
        z = torch.cat([z, y], 1)
        z = self.decoder_input(z)
        z = z.view(-1, 256, 2, 2)
        logit = self.model_decoder(z)
        feat = self.final_layer(logit)
        return feat

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()  # eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)  # z = eps*std+mu
        return z

    def forward(self, x, y):
        y = y.float()
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        x_rec = self.decode(z, y)
        return x_rec, mu, logvar
