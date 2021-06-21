import torch.nn as nn

class FSRCNN_x(nn.Module):
    def __init__(self, scale_factor, num_channels=3, d=23, s=12, m=2):
        super(FSRCNN_x, self).__init__()
        self.extract_features = nn.Sequential(nn.Conv2d(num_channels, d, kernel_size=3, padding=3//2), nn.ReLU())
        self.shrink = nn.Sequential(nn.Conv2d(d, s, kernel_size=1), nn.ReLU())
        self.map = []
        for _ in range(m):
            self.map.extend([nn.Conv2d(s, s, kernel_size=3, padding=3//2), nn.ReLU(s)])
        self.map = nn.Sequential(*self.map)
        self.expand = nn.Sequential(nn.Conv2d(s, d, kernel_size=1), nn.ReLU())
        self.deconv = nn.ConvTranspose2d(d, num_channels, kernel_size=3, stride=scale_factor, padding=3//2, output_padding=scale_factor-1)
    
    def forward(self, x):
        x = self.extract_features(x)
        x = self.shrink(x)
        x = self.map(x)
        x = self.expand(x)
        x = self.deconv(x)
        return x