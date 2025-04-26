# Modelos de rede

Modelos:

- Modelo de rede convolucional
    - nn.Conv2d
        - **torch.nn.Conv2d(*in_channels*, *out_channels*, *kernel_size*, *stride=1*, *padding=0*, *dilation=1*, *groups=1*, *bias=True*, *padding_mode='zeros'*, *device=None*, *dtype=None*)**
        - Aplica uma convolução 2D sobre um sinal de entrada composto por vários planos de entrada
        - In the simplest case, the output value of the layer with input size ( N , C in , H , W ) (N,C in  ,H,W) and output ( N , C out , H out , W out ) (N,C out  ,H out  ,W out  ) can be precisely described as:
        
        $$
        out(N_{t} C_{out_{j}}) = bias(C_{out_{j}}) + \sum_{k=0}^{c_{in}-1}weight(C_{out_{j}},K)*input(N_{i},k)
        $$
        
        - where * is the valid 2D [Cross-Correlation](https://en.wikipedia.org/wiki/Cross-correlation) operator, N is a batch size, C denotes a number of channels, H is a height of input planes in pixels, and W is width in pixels.
        - Exemplo:
            
            ```python
            # With square kernels and equal stride
            m = nn.Conv2d(16, 33, 3, stride=2)
            # non-square kernels and unequal stride and with padding
            m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
            # non-square kernels and unequal stride and with padding and dilation
            m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
            input = torch.randn(20, 16, 50, 100)
            output = m(input)
            
            ```
            
            - **`stride` controls the stride for the cross-correlation, a single number or a tuple.**
            - **`padding` controls the amount of padding applied to the input. It can be either a string {‘valid’, ‘same’} or an int / a tuple of ints giving the amount of implicit padding applied on both sides.**
            - **`dilation` controls the spacing between the kernel points; also known as the à trous algorithm. It is harder to describe, but this [link](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md) has a nice visualization of what `dilation` does**
            - **`groups` controls the connections between inputs and outputs. `in_channels` and `out_channels` must both be divisible by `groups`. For example,**
                - **At groups=1, all inputs are convolved to all outputs.**
                - **At groups=2, the operation becomes equivalent to having two conv layers side by side, each seeing half the input channels and producing half the output channels, and both subsequently concatenated.**
                - **At groups= `in_channels`, each input channel is convolved with its own set of filters (of size out_channelsin_channelsin_channelsout_channels).**
    
    ```python
    # Create a convolutional neural network
    class FashionMNISTModelV2(nn.Module):
        """
        Model architecture copying TinyVGG from:
        https://poloclub.github.io/cnn-explainer/
        """
        def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
            super().__init__()
            self.block_1 = nn.Sequential(
                nn.Conv2d(in_channels=input_shape,
                          out_channels=hidden_units,
                          kernel_size=3, # how big is the square that's going over the image?
                          stride=1, # default
                          padding=1),# options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number
                nn.ReLU(),
                nn.Conv2d(in_channels=hidden_units,
                          out_channels=hidden_units,
                          kernel_size=3,
                          stride=1,
                          padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2,
                             stride=2) # default stride value is same as kernel_size
            )
            self.block_2 = nn.Sequential(
                nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                # Where did this in_features shape come from?
                # It's because each layer of our network compresses and changes the shape of our input data.
                nn.Linear(in_features=hidden_units*7*7,
                          out_features=output_shape)
            )
    
        def forward(self, x: torch.Tensor):
            x = self.block_1(x)
            # print(x.shape)
            x = self.block_2(x)
            # print(x.shape)
            x = self.classifier(x)
            # print(x.shape)
            return x
      
    
    ```
    
- Multilayer perceptron
    
    Algortimo de classificação de imagem usando MLP - *Multilayer Perceptron*):
    
    ```python
    class MLPClasscifier(nn.Module):
      def __init__(self):
        super().__init__()
    
        #interpertrar a imagem(como um vetor), matriz de pixel
        self.flatten = nn.Flatten()
    
        #como vai trabalhar com os numeros q receber
        self.layers= nn.Sequential(
            nn.Linear(32 * 32 * 3, 256), #alturaxlagura(da imagem em pixels) * 3(RGB)
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            nn.ReLU()
        )
    
      def forward(self, x): #como o dado passa pela rede
        v = self.flatten(x)
        return self.layers(v)
    ```
    
- Torch.flatten():