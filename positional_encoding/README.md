# Positional Encoding

# PE Class

BARF Reference: [BARF/model/planar.py](https://github.com/chenhsuanlin/bundle-adjusting-NeRF/blob/99218cc543eca5453a7c4829638ccffd0a7ad2f5/model/planar.py#L232) 
NERF Reference: [BARF/model/nerf.py](https://github.com/chenhsuanlin/bundle-adjusting-NeRF/blob/363e077fa9627231bb9b4a69a143b2ef9467d7be/model/nerf.py#L410) 

```python
class PositionEncoding(torch.nn.Module):
    
    def __init__(self, barf_c2f, max_epochs):
        super().__init__()
        self.barf_c2f = barf_c2f
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.progress = torch.nn.Parameter(torch.tensor(0.)) # use Parameter so it could be checkpointed
        self.max_epochs = max_epochs
    
    def positional_encoding(self, input, current_epoch, L=8): # [B,...,N]
        shape = input.shape
        freq = 2**torch.arange(L,dtype=torch.float32, device=self.device)*np.pi # [L]
        spectrum = input[...,None]*freq # [B,...,N,L]
        sin,cos = spectrum.sin(),spectrum.cos() # [B,...,N,L]
        input_enc = torch.stack([sin,cos],dim=-2) # [B,...,N,2,L]
        input_enc = input_enc.view(*shape[:-1],-1) # [B,...,2NL]
        # coarse-to-fine: smoothly mask positional encoding for BARF
        self.progress = current_epoch/self.max_epochs
        if self.barf_c2f is not None:
            # set weights for different frequency bands
            start, end = self.barf_c2f
            alpha = (self.progress.data-start)/(end-start)*L
            k = torch.arange(L,dtype=torch.float32,device=self.device)
            weight = (1-(alpha-k).clamp_(min=0,max=1).mul_(np.pi).cos_())/2
            # apply weights
            shape = input_enc.shape
            input_enc = (input_enc.view(-1, L)*weight).view(*shape)
        return input_enc
```
# Usage

```python
if __name__ == "__main__":
    # Read config file
    with open("config.yaml", 'r') as file:
        cfg =yaml.safe_load(file)
    
    # max iterations
    max_epochs = 6
    
    # points shape = [B, HW, 3]
    input = torch.randn(10, 100, 3)
    
    # define class object
    position_encoding = PositionEncoding(cfg['barf_c2f'], max_epochs)
    
    # Inside training loop
    for epoch in range(max_epochs): 
        points_enc = position_encoding(input, epoch, cfg['planar']['posenc']['L_2D'])
        points_enc = torch.cat([input, points_enc],dim=-1) # [B,...,6L+3]
        
        # For 3D input samples = [B, HW, N, 3] 
        # points_enc = position_encoding(input, epoch, cfg['nerf']['posenc']['L_3D'])
        # points_enc = torch.cat([input, points_enc],dim=-1) # [B,...,6L+3]
```

3D Points definition Reference: [BARF/model/nerf.py](https://github.com/chenhsuanlin/bundle-adjusting-NeRF/blob/363e077fa9627231bb9b4a69a143b2ef9467d7be/model/nerf.py#L383) 


