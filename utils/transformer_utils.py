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