from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torchinfo import summary


class SegmentationMapFeatureExtractor(nn.Module):
    def __init__(self, num_classes=30, embedding_dim=8, use_embedding=True):
        super(SegmentationMapFeatureExtractor, self).__init__()
        self.use_embedding = use_embedding

        if self.use_embedding:
            # Embedding layer for class indices
            self.embedding = nn.Embedding(num_classes, embedding_dim)
        else:
            # For one-hot encoding, the feature size would be num_classes instead of embedding_dim
            self.embedding_dim = num_classes

        # Full-sized version of the architecture with larger channels
        self.conv1 = nn.Conv2d(embedding_dim if self.use_embedding else num_classes, 64, kernel_size=3, stride=2, padding=1)  # 32 -> 16
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # 16 -> 8
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # 8 -> 4
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)  # 4 -> 2
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)  # 2 -> 1

        # Global average pooling to convert the final output to 1x1024
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # ReLU for non-linearity
        self.relu = nn.ReLU()

    def _init_weights(self, dtype, device):
        '''
        The init in LLava is a bit weird and causes issues. Instead we will save a dict from a previously seperate run and use those to initialize. These weights are also random but init correctly
        :return:
        '''
        save_path = Path(f'llava/model/multimodal_projector/segmentation_map_feature_extractor_withembeddings{self.use_embedding}.pt')
        if save_path.exists():
            # load the weights
            print(f'Loading SegmentationMapFeatureExtractor weights from {save_path}')
            self.load_state_dict(torch.load(save_path), strict=True)

        else:
            # instead save there the current weights
            print(f'Saving current SegmentationMapFeatureExtractor weights to {save_path}')
            torch.save(self.state_dict(), save_path)

        # apply dtype and device
        self.to(dtype=dtype, device=device)

    def forward(self, x):
        # Input: x is expected to be of shape (batch_size, 1, 32, 32)
        assert x.shape[1:] == (32, 32), f"Expected input size (batch_size, 32, 32), but got {x.shape}"

        if self.use_embedding:
            # Use embedding: transform the class indices to embeddings
            x = self.embedding(x.long())  # Resulting in (32, 32, embedding_dim)
            x = x.permute(0, 3, 1, 2)  # Change to (batch_size, embedding_dim, 32, 32)
        else:
            # One-hot encoding, no hardcoded float type
            x = F.one_hot(x.long(), num_classes=30).to(x.dtype)  # (batch_size, 32, 32, num_classes)
            x = x.permute(0, 3, 1, 2)  # Change to (batch_size, num_classes, 32, 32)

        # Apply convolution layers with ReLU non-linearity
        x = self.relu(self.conv1(x))  # (batch_size, 64, 16, 16)
        x = self.relu(self.conv2(x))  # (batch_size, 128, 8, 8)
        x = self.relu(self.conv3(x))  # (batch_size, 256, 4, 4)
        x = self.relu(self.conv4(x))  # (batch_size, 512, 2, 2)
        x = self.relu(self.conv5(x))  # (batch_size, 1024, 1, 1)

        # Global average pooling to obtain a final output of (batch_size, 1024, 1, 1)
        x = x.squeeze(-1).squeeze(-1)  # Remove the last two dimensions
        return x


if __name__ == "__main__":
    # Example usage with assert for input size check
    segmentation_map_feature_extractor = SegmentationMapFeatureExtractor(input_dim=1, num_classes=30, embedding_dim=8, use_embedding=False)
    summary(segmentation_map_feature_extractor, col_names=['num_params', 'trainable'])
    segmentation_map_feature_extractor._init_weights(dtype=torch.bfloat16, device=torch.device('cuda'))
