import torch
import torch.nn as nn

class Patchification(nn.Module):
  """
  (a) Process the batch of images to non-overlapping patches
  Input shape: [batch, channel, height, width]
  Return: [batch, number_of_patches, embedding_dimension]
  """
  def __init__(self, in_channels, patch_size, embedding_dim):
    super().__init__()
    """
    Hint: embedding_dim should be the out_channel of convolution.
    """
    ##### YOUR CODE #####
    self.patchifying= nn.Conv2d(in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size[0])
    #####################

  def forward(self, x):
    ##### YOUR CODE #####
    x=self.patchifying(x)
    x=x.reshape(x.shape[0],x.shape[1],-1)
    x=x.transpose(1,2)
    #####################
    return x

class Linear_Patchification(nn.Module):
  """
  Extra credit.
  (b) Process the batch of images to non-overlapping patches
  Input shape: [batch, channel, height, width]
  Return: [batch, number_of_patches, embedding_dimension]
  """
  def __init__(self, in_channels, patch_size, embedding_dim):
    super().__init__()
    ##### YOUR CODE #####
    self.patch_size = patch_size
    self.patchifying= nn.Linear(patch_size[0]*patch_size[1]*in_channels, embedding_dim)
    #####################

  def forward(self, x):
    ##### YOUR CODE #####
    b = x.shape[0] # batch
    c = x.shape[1] # channel
    H = x.shape[2] # height
    W = x.shape[3] # width
    s1 = self.patch_size[0] # patch height
    h = int(H/s1)
    s2 = self.patch_size[1] # patch width
    w = int(W/s2)

    x = x.reshape(b,c,h,s1,w,s2)
    x = x.permute(0,2,4,3,5,1)
    x = x.reshape(b,h*w,s1*s2*c)
    x = self.patchifying(x)
    #####################
    return x

class MLP(nn.Module):
  """
  (c) Feed-forward layer
  Input shape: [batch, number_of_patches, embedding_dimension]
  Return: [batch, number_of_patches, embedding_dimension]
  """
  def __init__(self, dim, hidden_dim, dropout = 0.):
    super().__init__()
    ##### YOUR CODE #####
    self.mlp_layer = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim,dim)
    )
    #####################

  def forward(self, x):
    ##### YOUR CODE #####
    residual = x
    x = self.mlp_layer(x)
    x = x + residual
    #####################
    return x

class Attention(nn.Module):
  """
  (d) Multi-head attention
  You can implement without considering heads (i.e. vanilla attention).
  However, your maximum score will be 8 points.

  Input shape: [batch, number_of_patches, embedding_dimension]
  Return: [batch, number_of_patches, embedding_dimension]
  """
  def __init__(self, dim, num_heads, dropout = 0.):
    super().__init__()
    """
    Do NOT modify.
    """
    self.head_dim = dim // num_heads
    self.dim = dim
    self.num_heads = num_heads

    self.scale = self.head_dim ** 0.5 # Don't forget scaling!
    # If you are not going to consider head, you need to change self.scale as below.
    # self.scale = self.dim ** 0.5

    self.dropout = nn.Dropout(dropout)

    ##### YOUR CODE #####
    """
    You need to initialize some layers...
    """
    self.to_key = nn.Linear(dim, dim)
    self.to_query = nn.Linear(dim, dim)
    self.to_value = nn.Linear(dim, dim)
    self.to_out = nn.Linear(dim, dim)
    #####################

  def forward(self, x):
    ##### YOUR CODE #####
    batch_size= x.shape[0]
    query = self.to_query(x)
    key = self.to_key(x)
    value = self.to_value(x)

    query =query.reshape(batch_size,-1,self.num_heads, self.head_dim).transpose(1,2)   #[batch, num_patches, num_heads*head_dim]-> [batch, num_heads, num_patches, head_dim]
    key =key.reshape(batch_size,-1,self.num_heads, self.head_dim).transpose(1,2)
    value =value.reshape(batch_size,-1,self.num_heads, self.head_dim).transpose(1,2)

    attn_score = torch.matmul(query, key.transpose(-2, -1)) /self.scale
    attn_coef=torch.softmax(attn_score, dim=-1)
    attn_coef= self.dropout(attn_coef)

    attn = torch.matmul(attn_coef, value).transpose(1, 2) # [batch, num_head, num_patches, head_dim]->[batch, num_pathces,num_head, head_dim]
    attn= attn.reshape(attn.shape[0],attn.shape[1],-1) # ->[batch, num_pathces,num_head*head_dim]
    x= self.to_out(attn)
    #####################
    return x

class Block(nn.Module):
  """
  (e) Attention block
  Input shape: [batch, number_of_patches, embedding_dimension]
  Return: [batch, number_of_patches, embedding_dimension]
  """
  def __init__(self, dim, num_heads, mlp_dim, dropout=0.):
    super().__init__()
    ##### YOUR CODE #####
    self.block1 = nn.Sequential(
      nn.LayerNorm(dim),
      Attention(dim, num_heads, dropout),
      nn.Dropout(dropout)
    )

    self.block2 = nn.Sequential(
      nn.LayerNorm(dim),
      MLP(dim, mlp_dim, dropout),
      nn.Dropout(dropout)
    )

    #####################

  def forward(self, x):
    """
    Hint: Don't forget the residual connections!
    (Refer the lab3 slide)
    """
    ##### YOUR CODE #####
    residual1 = x
    x = self.block1(x)
    x = x + residual1

    residual2 = x
    x = self.block2(x)
    x = x + residual2

    #####################
    return x


class ViT(nn.Module):
    def __init__(self, image_shape, patch_size, num_classes, dim, num_heads, depth, mlp_dim, dropout = 0.):
        super().__init__()
        """
        image_shape: [channel, height, width]
        patch_size: [height, width]
        dim: Embedding dimension
        num_heads: Number of heads to be used in Multi-head Attention
        depth: Number of attention blocks to be used
        mlp_dim: Hidden dimension to be used in MLP layer (=feedforward layer)
        """

        image_ch, image_h, image_w = image_shape # image_ch will be 3(RGB 3 channels) for CIFAR10 dataset
        patch_h, patch_w = patch_size

        assert image_h % patch_h == 0 and image_w % patch_w == 0, 'Image height & width must be divisible by those of patch respectively.'
        assert dim % num_heads == 0, 'Embedding dimension should be divisible by number of heads.'

        num_patches = (image_h // patch_h) * (image_w // patch_w) # e.g. [32 x 32] image & [8 x 8] patch size -> [4 x 4 = 16] patches

        # Patchification using convolution.
        self.patchify = Patchification(image_ch, patch_size, dim)

        # Linear Patchification for extra credit.
        # self.patchify = Linear_Patchification(image_ch, patch_size, dim)

        # Learnable positional encoding, 1+ is for class token.
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + num_patches, dim))

        # Class token which will be prepended to each image.
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # Initialize attention blocks
        self.attention_blocks = nn.ModuleList([
            Block(dim, num_heads, mlp_dim, dropout)
            for _ in range(depth)
        ])

        # Classification head, maps the final vector to class dimension.
        self.classification_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1) # Shape: [batch, 1, dim]
        """
        (f) ViT forward
        Hint!
        After patchification, shape will be [batch, number_of_patches, dim].
        If you successfully prepend cls_tokens to this batch of patchfied images, shape will be [batch, 1+ number_of_patches, dim].
        Then simply add the positional embedding.
        Now the tokens(patches) are ready to go through the attention blocks.
        After attention operation, classify with class token. (Simply take off it from whole tokens)
        """
        ##### YOUR CODE #####
        x = self.patchify(img)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embedding

        for attention_block in self.attention_blocks:
          x = attention_block(x)

        x = x[:,0,:]
        x = self.classification_head(x)

        #####################
        return x