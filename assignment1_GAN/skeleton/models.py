# Assignment 1 - GAN Playground
# models.py  --  Architecture definitions
#
# Classes to implement:
#   - DCGenerator        (Part 1)
#   - DCDiscriminator    (Part 1 & 2)
#   - CycleGenerator     (Part 2)
#   - PatchDiscriminator (Part 2)

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helper builders (provided – do not modify)
# ---------------------------------------------------------------------------

def up_conv(in_channels, out_channels, kernel_size, stride=1, padding=1,
            scale_factor=2, norm='batch', activ=None):
    """Upsample then Conv2d, with optional normalisation and activation.

    Set scale_factor=1 to skip the upsample step (useful for the first
    generator layer that maps 1x1 noise -> 4x4 feature map).
    """
    layers = []
    layers.append(nn.Upsample(scale_factor=scale_factor, mode='nearest'))
    layers.append(nn.Conv2d(in_channels, out_channels,
                            kernel_size, stride, padding, bias=(norm is None)))
    if norm == 'batch':
        layers.append(nn.BatchNorm2d(out_channels))
    elif norm == 'instance':
        layers.append(nn.InstanceNorm2d(out_channels))
    if activ == 'relu':
        layers.append(nn.ReLU())
    elif activ == 'leaky':
        layers.append(nn.LeakyReLU())
    elif activ == 'tanh':
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


def conv(in_channels, out_channels, kernel_size, stride=2, padding=1,
         norm='batch', init_zero_weights=False, activ=None):
    """Strided Conv2d, with optional normalisation and activation."""
    layers = []
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                           kernel_size=kernel_size, stride=stride,
                           padding=padding, bias=(norm is None))
    if init_zero_weights:
        conv_layer.weight.data = 0.001 * torch.randn(
            out_channels, in_channels, kernel_size, kernel_size)
    layers.append(conv_layer)
    if norm == 'batch':
        layers.append(nn.BatchNorm2d(out_channels))
    elif norm == 'instance':
        layers.append(nn.InstanceNorm2d(out_channels))
    if activ == 'relu':
        layers.append(nn.ReLU())
    elif activ == 'leaky':
        layers.append(nn.LeakyReLU())
    elif activ == 'tanh':
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class ResnetBlock(nn.Module):
    """Single residual block used inside CycleGenerator (provided)."""

    def __init__(self, conv_dim, norm, activ):
        super().__init__()
        self.conv_layer = conv(in_channels=conv_dim, out_channels=conv_dim,
                               kernel_size=3, stride=1, padding=1,
                               norm=norm, activ=activ)

    def forward(self, x):
        return x + self.conv_layer(x)


# ---------------------------------------------------------------------------
# Part 1 – DCGAN
# ---------------------------------------------------------------------------

class DCDiscriminator(nn.Module):
    """Discriminator: maps a 64x64 RGB image -> scalar real/fake score.

    Architecture (x: (BS, 3, 64, 64)):
        conv1  ->  (BS, 32, 32, 32)  InstanceNorm  ReLU
        conv2  ->  (BS, 64, 16, 16)  InstanceNorm  ReLU
        conv3  ->  (BS, 128, 8, 8)   InstanceNorm  ReLU
        conv4  ->  (BS, 256, 4, 4)   InstanceNorm  ReLU
        conv5  ->  (BS, 1, 1, 1)     no norm       no activation
    """

    def __init__(self, conv_dim=64, norm='instance', leaky=False):
        super().__init__()

        activ = 'leaky' if leaky else 'relu'

        self.conv1 = conv(3,             conv_dim // 2, kernel_size=4, norm=norm, activ=activ)
        self.conv2 = conv(conv_dim // 2, conv_dim,      kernel_size=4, norm=norm, activ=activ)
        self.conv3 = conv(conv_dim,      conv_dim * 2,  kernel_size=4, norm=norm, activ=activ)
        self.conv4 = conv(conv_dim * 2,  conv_dim * 4,  kernel_size=4, norm=norm, activ=activ)
        self.conv5 = conv(conv_dim * 4,  1, kernel_size=4, stride=1, padding=0,
                          norm=None, activ=None)

    def forward(self, x):
        """
        Input
        -----
            x: (BS, 3, 64, 64)

        Output
        ------
            out: (BS, 1, 1, 1)  scalar score per image
        """
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        return out

class DCGenerator(nn.Module):
    """Generator: maps a noise vector z -> 64x64 RGB image.

    Architecture (z: (BS, noise_size, 1, 1)):
        up_conv1  ->  (BS, 256, 4, 4)   InstanceNorm  ReLU
        up_conv2  ->  (BS, 128, 8, 8)   InstanceNorm  ReLU
        up_conv3  ->  (BS, 64, 16, 16)  InstanceNorm  ReLU
        up_conv4  ->  (BS, 32, 32, 32)  InstanceNorm  ReLU
        up_conv5  ->  (BS, 3, 64, 64)   no norm       Tanh
    """

    def __init__(self, noise_size, conv_dim=64):
        super().__init__()

        # 1x1 noise -> 4x4 feature map: skip the upsample and use a
        # padded conv to expand the spatial size.
        self.up_conv1 = up_conv(noise_size,   conv_dim * 4, kernel_size=4,
                                stride=1, padding=3, scale_factor=1,
                                norm='instance', activ='relu')

        self.up_conv2 = up_conv(conv_dim * 4, conv_dim * 2, kernel_size=3,
                                norm='instance', activ='relu')

        self.up_conv3 = up_conv(conv_dim * 2, conv_dim,     kernel_size=3,
                                norm='instance', activ='relu')

        self.up_conv4 = up_conv(conv_dim,     conv_dim // 2, kernel_size=3,
                                norm='instance', activ='relu')

        self.up_conv5 = up_conv(conv_dim // 2, 3, kernel_size=3,
                                norm=None, activ='tanh')

    def forward(self, z):
        """
        Input
        -----
            z: (BS, noise_size, 1, 1)

        Output
        ------
            out: (BS, channels, image_width, image_height)
        """
        out = self.up_conv1(z)
        out = self.up_conv2(out)
        out = self.up_conv3(out)
        out = self.up_conv4(out)
        out = self.up_conv5(out)
        return out


# ---------------------------------------------------------------------------
# Part 2 – CycleGAN
# ---------------------------------------------------------------------------

class CycleGenerator(nn.Module):
    """Encoder–ResNet–Decoder generator for CycleGAN.

    Architecture (x: (BS, 3, 64, 64)):
        Encoder
            conv1   ->  (BS, 32, 32, 32)  InstanceNorm  ReLU
            conv2   ->  (BS, 64, 16, 16)  InstanceNorm  ReLU
        Transform
            resnet_block x 3  ->  (BS, 64, 16, 16)  InstanceNorm  ReLU
        Decoder
            up_conv1  ->  (BS, 32, 32, 32)  InstanceNorm  ReLU
            up_conv2  ->  (BS,  3, 64, 64)  (no norm)     Tanh
    """

    def __init__(self, conv_dim=64, init_zero_weights=False, norm='instance'):
        super().__init__()

        # Encoder
        self.conv1 = conv(3,             conv_dim // 2, kernel_size=4,
                          norm=norm, activ='relu',
                          init_zero_weights=init_zero_weights)
        self.conv2 = conv(conv_dim // 2, conv_dim,      kernel_size=4,
                          norm=norm, activ='relu',
                          init_zero_weights=init_zero_weights)

        # Transform (3 residual blocks)
        self.resnet_block = nn.Sequential(*[
            ResnetBlock(conv_dim, norm=norm, activ='relu') for _ in range(3)
        ])

        # Decoder
        self.up_conv1 = up_conv(conv_dim,      conv_dim // 2, kernel_size=3,
                                norm=norm, activ='relu')
        self.up_conv2 = up_conv(conv_dim // 2, 3, kernel_size=3,
                                norm=None, activ='tanh')

    def forward(self, x):
        """
        Input
        -----
            x: (BS, 3, 64, 64)

        Output
        ------
            out: (BS, 3, 64, 64)
        """
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.resnet_block(out)
        out = self.up_conv1(out)
        out = self.up_conv2(out)
        return out


class PatchDiscriminator(nn.Module):
    """Patch-based discriminator for CycleGAN.

    Produces a spatial output (e.g. 4x4) rather than a scalar, so the
    loss is computed patch-wise.

    Hint: this is very similar to DCDiscriminator – you essentially remove
    one layer so the spatial dimensions are not collapsed all the way to 1x1.
    """

    def __init__(self, conv_dim=64, norm='instance'):
        super().__init__()

        self.conv1 = conv(3,             conv_dim // 2, kernel_size=4, norm=norm, activ='relu')
        self.conv2 = conv(conv_dim // 2, conv_dim,      kernel_size=4, norm=norm, activ='relu')
        self.conv3 = conv(conv_dim,      conv_dim * 2,  kernel_size=4, norm=norm, activ='relu')
        self.conv4 = conv(conv_dim * 2,  conv_dim * 4,  kernel_size=4, norm=norm, activ='relu')
        # Keep the spatial size at 4x4 so we get patch-level scores.
        self.conv5 = conv(conv_dim * 4,  1, kernel_size=3, stride=1, padding=1,
                          norm=None, activ=None)

    def forward(self, x):
        """
        Input
        -----
            x: (BS, 3, 64, 64)

        Output
        ------
            out: (BS, 1, 4, 4)  patch-level scores
        """
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        return out


# ---------------------------------------------------------------------------
# Quick shape test (run: python models.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import torch
    x = torch.rand(4, 3, 64, 64)
    z = torch.rand(4, 100, 1, 1)

    def run_shape_test(name, build_fn, input_tensor, expected_shapes):
        print(f"=== {name} ===")
        try:
            model = build_fn()
            output = model(input_tensor)
            actual_shape = tuple(output.shape)
            expected_ok = actual_shape in expected_shapes
            if expected_ok:
                print(f"PASS: output shape {output.shape}")
            else:
                expected_str = " or ".join(str(s) for s in expected_shapes)
                print(
                    f"FAIL: output shape {output.shape}, expected {expected_str}"
                )
        except NotImplementedError as e:
            print(f"not implemented: {e}")
        except Exception as e:
            print(f"failed: {type(e).__name__}: {e}")

    run_shape_test("PatchDiscriminator", PatchDiscriminator, x, {(4, 1, 4, 4)})
    print()

    run_shape_test("CycleGenerator", CycleGenerator, x, {(4, 3, 64, 64)})
    print()

    run_shape_test(
        "DCGenerator",
        lambda: DCGenerator(noise_size=100),
        z,
        {(4, 3, 64, 64)},
    )
    print()

    run_shape_test("DCDiscriminator", DCDiscriminator, x, {(4, 1, 1, 1)})
