
# import the necessary packages
from torch import nn

class CNN(nn.Module):
    def __init__(self):     
        super(CNN,self).__init__()
        self.conv2d_1 = nn.Conv2d( 
            out_channels = 8,
            kernel_size = 3,
            stride = 1,
            padding = 1,
            dilation = 1,
            groups = 1,
            bias = True,
            padding_mode = "zeros",
            in_channels = 3,
        )
        
        self.relu_1 = nn.ReLU( 
            inplace = False,
        )
        
        self.conv2d_2 = nn.Conv2d( 
            out_channels = 16,
            kernel_size = 3,
            stride = 1,
            padding = 1,
            dilation = 1,
            groups = 1,
            bias = True,
            padding_mode = "zeros",
            in_channels = 8,
        )
        
        self.relu_2 = nn.ReLU( 
            inplace = False,
        )
        
        self.conv2d_3 = nn.Conv2d( 
            out_channels = 32,
            kernel_size = 3,
            stride = 1,
            padding = 1,
            dilation = 1,
            groups = 1,
            bias = True,
            padding_mode = "zeros",
            in_channels = 16,
        )
        
        self.relu_3 = nn.ReLU( 
            inplace = False,
        )
        
        self.flatten_1 = nn.Flatten( 
            start_dim = 0,
            end_dim = -1,
        )
        
        self.linear_1 = nn.Linear( 
            out_features = 100,
            bias = True,
            in_features = 13117440,
        )
        
        self.linear_2 = nn.Linear( 
            out_features = 2,
            bias = True,
            in_features = 100,
        )
        

    def forward(self, x):
        x = self.conv2d_1(x)
        x = self.relu_1(x)
        x = self.conv2d_2(x)
        x = self.relu_2(x)
        x = self.conv2d_3(x)
        x = self.relu_3(x)
        x = self.flatten_1(x)
        x = self.linear_1(x)
        x = self.linear_2(x)
        
        return x