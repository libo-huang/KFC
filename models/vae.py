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
    """
    Conditional variational autoencoder network used in LGLvKR,
    Especially for the MNIST and fashion-MNIST datasets.
    """
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
        # Recommend using Sigmoid() instead on MNIST since its pixel values are between 0 and 1.
        self.sigmoid = nn.LeakyReLU()

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
    """
    Conditional variational autoencoder network used in LGLvKR,
    Especially for the SVHN and CIFAR-10 datasets.
    """
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

class ConditionalVAE_cifar(nn.Module):
    def __init__(self, args):
        super(ConditionalVAE_cifar, self).__init__()
        self.latent_dim = args.latent_dim
        self.img_size = 64
        in_channels = 3 if args.dataset == 'cifar10' or args.dataset == 'svhn' else 1

        self.embed_class = nn.Linear(args.class_dim, self.img_size * self.img_size)
        self.embed_data = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        modules = []
        # if hidden_dims is None:
        hidden_dims = [32, 64, 128, 256, 512]

        in_channels += 1 # To account for the extra label channel
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, self.latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, self.latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(self.latent_dim + args.class_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return mu, log_var

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, y):
        # y = y.reshape(1,y.size(0)).transpose(1,0).squeeze()
        # y = torch.eye(10)[y].cuda()
        y = y.float()
        embedded_class = self.embed_class(y)
        embedded_class = embedded_class.view(-1, self.img_size, self.img_size).unsqueeze(1)
        embedded_input = self.embed_data(input)

        x = torch.cat([embedded_input, embedded_class], dim = 1)
        mu, log_var = self.encode(x)

        z = self.reparameterize(mu, log_var)

        z = torch.cat([z, y], dim = 1)
        return self.decode(z), mu, log_var
