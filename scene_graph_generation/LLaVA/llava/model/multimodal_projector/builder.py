import re

import torch
import torch.nn as nn
from llava.model.multimodal_projector.pointtransformerv3 import Point
from llava.model.multimodal_projector.pointtransformerv3 import PointTransformerV3
from llava.model.multimodal_projector.segmentation_map_feature_extractor import SegmentationMapFeatureExtractor
from torch.cuda import amp
from transformers import BertConfig, BertModel


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )

    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')


class ImageEmbeddingPooler(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_dim = 1024

        # Configure a new BERT model with 2 hidden layers and without positional embeddings
        # used for processing images, regardless of azure, simstation or trackercam
        config = BertConfig(
            hidden_size=self.embedding_dim,
            num_hidden_layers=2,  # Set the number of hidden layers to 2
            num_attention_heads=8,
            intermediate_size=self.embedding_dim * 4,
            use_position_embeddings=True,
            # max_position_embeddings=2304, # TODO adjust as needed
            # max_position_embeddings=2880,
            max_position_embeddings=576 * 7,  # max we will have 10 images. But with batchsize 4 only 8.
            use_bfloat16=True,
            vocab_size=1,
        )
        self.bert = BertModel(config)

        # used for processing point clouds
        self.point_transformer = PointTransformerV3(
            cls_mode=True,
            project_pc_dim=1024
        )
        self.point_pooling = nn.AdaptiveAvgPool1d(1)  # Global Average Pooling to reduce to 1xD

        self.project_audio = nn.Linear(512, 1024)

        self.segmasks_encoder = SegmentationMapFeatureExtractor(num_classes=30, embedding_dim=8, use_embedding=True)

    def _encode_pc(self, point_clouds):
        device = torch.device('cuda')
        self.point_transformer.float()
        real_batch_size = len(point_clouds)

        # Pre-initialize pooled_feats with zero tensors for all entries
        pc_feats = torch.zeros((real_batch_size, 512), dtype=torch.float, device=device)  # TODO this needs to be kept up to date

        batch = []
        all_coords = []
        all_feats = []
        valid_batch_idx = 0  # Continuous batch indices for valid point clouds
        for i in range(real_batch_size):
            point_cloud = point_clouds[i]
            if point_cloud is None:
                # If point cloud is None, leave the corresponding pc_feats[i] as zero
                continue
            point_cloud = point_cloud.float().to(device)
            num_points = point_cloud.shape[0]

            # Use valid_batch_idx for continuous batch indices for valid point clouds
            batch.append(torch.full((num_points,), valid_batch_idx, dtype=torch.long, device=device))
            valid_batch_idx += 1
            all_coords.append(point_cloud[:, :3])  # xyz coordinates
            all_feats.append(point_cloud)  # xyzrgb features

        if len(batch) > 0:
            # Concatenate all valid point cloud data
            batch = torch.cat(batch, dim=0)
            all_coords = torch.cat(all_coords, dim=0)
            all_feats = torch.cat(all_feats, dim=0)
            point_data = Point(
                coord=all_coords,  # xyz coordinates
                feat=all_feats,  # xyzrgb features
                grid_size=torch.tensor(0.01, dtype=torch.float, device=device),  # Grid size for voxelization
                batch=batch  # Continuous batch indices
            )
            # Pass valid point cloud data through the point_transformer
            point_data = self.point_transformer(point_data)
            feat = point_data.feat
            # Overwrite the zero-initialized pc_feats with valid point cloud features
            valid_batch_idx = 0
            for i in range(real_batch_size):
                point_cloud = point_clouds[i]
                if point_cloud is None:
                    continue  # Skip None point clouds, leave zero tensor in pooled_feats[i]
                mask = point_data['batch'] == valid_batch_idx
                pooled_feat = self.point_pooling(feat[mask].unsqueeze(0).permute(0, 2, 1)).squeeze(-1)
                # Now fill the corresponding pc_feats[i] (original index) with the computed value
                pc_feats[i] = pooled_feat
                valid_batch_idx += 1

        # Project all features (including zeros) through the final projection layer
        pc_feats = self.point_transformer.project_pc(pc_feats.float())

        return pc_feats

    def _encode_audio(self, audios, length):
        # we assume there are already proccessed embeddings, computed in create_take_sample_audio_embeddings.py. But still handle the case of necessary linear projections. and zeros.
        audio_feats = torch.zeros((length, 512), dtype=torch.bfloat16, device=torch.device('cuda'))
        if audios is not None:
            for i, audio in enumerate(audios):
                if audio is not None:
                    audio_feats[i] = audio.to(torch.bfloat16)

        audio_feats = self.project_audio(audio_feats)
        return audio_feats

    def _encode_segmasks(self, segmasks):
        # we assume 3 masks per sample
        segmasks_feats = torch.zeros((len(segmasks), 3, 1024), dtype=torch.bfloat16, device=torch.device('cuda'))
        for i, segmask in enumerate(segmasks):
            if segmask is not None:
                segmasks_feats[i, :len(segmask)] = self.segmasks_encoder(torch.stack(segmask).to(torch.device('cuda')).to(torch.bfloat16))
        return segmasks_feats

    def forward(self, embeddings, attention_mask, pc=None, audio=None, segmasks=None):
        # embeddings shape: (batch_size, num_images, embedding_dim)
        batch_size, num_tokens, _ = embeddings.shape
        # Process embeddings through BERT without positional IDs
        outputs = self.bert(inputs_embeds=embeddings, attention_mask=attention_mask)
        # identity option
        outputs = outputs['last_hidden_state'].to(embeddings.dtype)[:, :576]
        with amp.autocast(enabled=False):
            pc_outputs = self._encode_pc(pc) if pc is not None else None
        pc_outputs = pc_outputs.to(outputs.dtype) if pc_outputs is not None else None
        audio_outputs = self._encode_audio(audio, length=len(outputs)) if audio is not None else None
        segmasks_outputs = self._encode_segmasks(segmasks) if segmasks is not None else None
        features_to_add = []
        if pc_outputs is not None:
            features_to_add.append(pc_outputs.unsqueeze(1))
        if audio_outputs is not None:
            features_to_add.append(audio_outputs.unsqueeze(1))
        if segmasks_outputs is not None:
            features_to_add.extend(segmasks_outputs.transpose(0, 1).unsqueeze(2))
        if len(features_to_add) > 0:
            outputs = torch.cat([outputs] + features_to_add, dim=1)
        return outputs


def build_image_pooler(config):
    return ImageEmbeddingPooler()
