import  math
import  torch
import  torch.nn as nn
import  torch.nn.functional as F
import  torch.fft as fft
from    einops import rearrange
from    ops.pixelshuffle import pixelshuffle_block
from    ops.layernorm import LayerNorm2d

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x)) + x

class Conv_PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm2d(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x)) + x
    
class Gated_Conv_FeedForward(nn.Module):
    def __init__(self, dim, mult = 1, bias=False, dropout = 0.):
        super().__init__()

        hidden_features = int(dim*mult)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
    
class Attention(nn.Module):
    def __init__(self,num_in_ch=3, dim=64, num_heads=8, bias=False):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(num_in_ch, dim , kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(num_in_ch, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        # q = self.q_dwconv(self.q(x))
        q = x
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1) 

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out + x

class CrossAttention(nn.Module):
    def __init__(self,num_in_ch=3, dim=64, num_heads=8, p_size=32, bias=False):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(num_in_ch, dim , kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(num_in_ch, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        # self.prompt = nn.Parameter(torch.rand(1, dim, p_size, p_size))

    def forward(self, x, y):
        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(x))
        # q = q*self.prompt
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1) 
        # k = k*self.prompt
        # v = v*self.prompt

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out + x

  
class Channel_Attention(nn.Module):
    def __init__(
        self, 
        dim, 
        heads, 
        bias=False, 
        dropout = 0.,
        window_size = 8
    ):
        super(Channel_Attention, self).__init__()
        self.heads = heads

        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
       
        self.ps = window_size

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.chunk(3, dim=1) 

        q,k,v = map(lambda t: rearrange(t, 'b (head d) (h ph) (w pw) -> b (h w) head d (ph pw)', ph=self.ps, pw=self.ps, head=self.heads), qkv)
        
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature 
        attn = attn.softmax(dim=-1)
        out =  (attn @ v)

        out = rearrange(out, 'b (h w) head d (ph pw) -> b (head d) (h ph) (w pw)', h=h//self.ps, w=w//self.ps, ph=self.ps, pw=self.ps, head=self.heads)


        out = self.project_out(out)

        return out

class MSCA(nn.Module):
    def __init__(self, dim):
        super(MSCA, self).__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)
        
        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)

        attn = attn + attn_0 + attn_1 + attn_2

        attn = self.conv3(attn)

        return attn * u
    
class MSCAN(nn.Module):
    def __init__(self, dim):
        super(MSCAN, self).__init__()
        self.dim = dim
        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = MSCA(dim)
        self.proj_2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x
    
class Block(nn.Module):
    def __init__(self, num_in_ch=3, dim=64, num_heads = 4, window_size=8, dropout=0.0):
        super(Block, self).__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.shallow_feature = nn.Conv2d(num_in_ch,dim,1)
        self.spatial_feature = MSCAN(dim)
        self.freq_feat_conv1 = nn.Conv2d(dim, dim , kernel_size=1, bias=False)
        self.freq_feat_conv2 = nn.Conv2d(dim, dim , kernel_size=1, bias=False)
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=False)
        self.kv_conv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=1, groups=dim * 2, bias=False)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.attention = Attention(num_in_ch=dim, dim=dim, num_heads=num_heads)
        self.layer = nn.Sequential(
            #spatial attention
            Conv_PreNormResidual(dim, Gated_Conv_FeedForward(dim = dim, dropout=dropout)),
            # channel-like attention
            Conv_PreNormResidual(dim, Channel_Attention(dim = dim, heads=4, dropout = dropout, window_size = window_size)),
            Conv_PreNormResidual(dim, Gated_Conv_FeedForward(dim = dim, dropout = dropout))
        )
    
    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'constant', 0)
        return x

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.check_image_size(x)
            
        sh_f = self.shallow_feature(x)

        s_f = self.spatial_feature(sh_f)

        x_fft = fft.fftn(sh_f, dim=(-2, -1)).real
        x_fft1=self.freq_feat_conv1(x_fft)
        x_fft2=F.gelu(x_fft1)
        x_fft3=self.freq_feat_conv2(x_fft2)
        q = fft.ifftn(x_fft3,dim=(-2, -1)).real

        out = self.attention(q,s_f)
        out = self.layer(out)
        # k, v = self.kv_conv(self.kv(s_f)).chunk(2, dim=1)
        # q = q.reshape(b, self.num_heads, -1, h * w)
        # k = k.reshape(b, self.num_heads, -1, h * w)
        # v = v.reshape(b, self.num_heads, -1, h * w)
        # q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        # attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        # out = self.project_out(torch.matmul(attn, v).reshape(b, -1, h, w))
        out = out + sh_f
        out = out[:,:,:h,:w]
        return out

class UIE(nn.Module):
    def __init__(self,num_in_ch=3,num_out_ch=3,num_feat=64,**kwargs):
        super(UIE, self).__init__()
        self.up_scale    = kwargs["upsampling"]
        bias        = kwargs["bias"]
        self.window_size = kwargs["window_size"]
        self.block_00 = Block(num_in_ch=num_in_ch, dim=num_feat, num_heads=num_feat // 8)
        self.block_01 = Block(num_in_ch=num_feat, dim=num_feat, num_heads=num_feat // 8)
        self.block_10 = Block(num_in_ch=num_in_ch, dim=num_feat, num_heads=num_feat // 8)
        self.block_11 = Block(num_in_ch=num_feat, dim=num_feat, num_heads=num_feat // 8)
        self.block_20 = Block(num_in_ch=num_in_ch, dim=num_feat, num_heads=num_feat // 8)
        self.block_21 = Block(num_in_ch=num_feat, dim=num_feat, num_heads=num_feat // 8)

        self.fuse_1 = CrossAttention(num_in_ch=num_feat,dim=num_feat,p_size=32)
        self.fuse_2 = CrossAttention(num_in_ch=num_feat,dim=num_feat,p_size=64)

        self.up_20 = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
        self.up_21 = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
        self.up_10 = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
        self.up_11 = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
        self.up     = pixelshuffle_block(num_feat,num_out_ch,self.up_scale,bias=bias)
        self.input_upscale = nn.Upsample(scale_factor=self.up_scale, mode='bicubic', align_corners=True)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'constant', 0)
        return x

    def forward(self, x):
        input_upscale = self.input_upscale(x)

        H, W = x.shape[2:]
        x = self.check_image_size(x)
        # print(f"x shape : {x.shape}")
        down_1 = F.interpolate(x, scale_factor=0.5, mode="bicubic")
        # print(f"down_1 shape : {down_1.shape}")
        down_2 = F.interpolate(down_1, scale_factor=0.5, mode="bicubic")
        # print(f"down_2 shape : {down_2.shape}")

        down_00 = self.block_00(x)
        # print(f"down_00 shape : {down_00.shape}")
        down_10 = self.block_10(down_1)
        # print(f"down_10 shape : {down_10.shape}")
        down_20 = self.block_20(down_2)
        # print(f"down_20 shape : {down_20.shape}")
        # print("1")
        fuse_120 = self.fuse_1(self.up_20(down_20), down_10)
        # print(f"fuse_120 shape : {fuse_120.shape}")
        # print("2")
        fuse_010 = self.fuse_2(self.up_10(fuse_120), down_00)
        # print(f"fuse_010 shape : {fuse_010.shape}")
        
        down_01 = self.block_01(fuse_010)
        # print(f"down_01 shape : {down_01.shape}")
        down_11 = self.block_11(fuse_120)
        # print(f"down_11 shape : {down_11.shape}")
        down_21 = self.block_21(down_20)
        # print(f"down_21 shape : {down_21.shape}")
        # print("3")
        fuse_121 = self.fuse_1(self.up_21(down_21), down_11)
        # print(f"fuse_121 shape : {fuse_121.shape}")
        # print("4")
        fuse_011 = self.fuse_2(self.up_20(fuse_121), down_01)
        # print(f"fuse_011 shape : {fuse_011.shape}")
        # exit(0)

        out     = self.up(fuse_011)
        out = out[:, :, :H*self.up_scale, :W*self.up_scale]
        return out + input_upscale