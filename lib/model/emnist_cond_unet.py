import torch
import torch.nn as nn
from lib.model.layers import nonlinearity, get_timestep_embedding, ColorCycleEmbedding

from .unet import UNet


class ConditionalEMNISTUNet(UNet):

    def __init__(self,
                 data_config: dict,
                 model_config: dict,
                 diffusion_config: dict,
                 num_classes: int = None,
                
                 ):

        super(ConditionalEMNISTUNet, self).__init__(data_config, model_config, diffusion_config)

        #Embedding for phase
        if num_classes is not None:
            self.label_embedding = nn.Embedding(num_embeddings=num_classes, embedding_dim=self.temb_ch)


        #Embedding for colorCycle
        self.colorcycle_label_embedding = ColorCycleEmbedding(self.ch)
        self.colorcycle_label_embedding_dense = nn.ModuleList([
            torch.nn.Linear(self.ch,
                            self.temb_ch),
            torch.nn.Linear(self.temb_ch,
                            self.temb_ch),
        ])


        #Embedding for tool
        # if num_semantic_labels is not None:
        #     self.semantic_label_embedding = nn.ModuleList([
        #         torch.nn.Linear(num_semantic_labels,
        #                         self.temb_ch),
        #         torch.nn.Linear(self.temb_ch,
        #                         self.temb_ch),
        #     ])

    def forward(self, x, t, mask=None, label=None, current_color_cycle=None):

        assert x.shape[2] == x.shape[3] == self.data_shape[-1]

        # Timestep embedding
        time_emb = get_timestep_embedding(t, self.ch)
        time_emb = self.temb.dense[0](time_emb)
        time_emb = nonlinearity(time_emb)
        time_emb = self.temb.dense[1](time_emb)

        # Append label embedding to timestep embedding
        if label is not None:
            # Reshaping phase label embedding to one dim
            time_emb += self.label_embedding(label.squeeze(-1))

        #Append colorcycle embedding to prev embedding
        if current_color_cycle is not None:
            cycle_emb = self.colorcycle_label_embedding(current_color_cycle.squeeze())
            cycle_emb = self.colorcycle_label_embedding_dense[0](cycle_emb)
            nonlinearity(cycle_emb)
            cycle_emb = self.colorcycle_label_embedding_dense[1](cycle_emb)
            time_emb += cycle_emb

        # Downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], time_emb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # Bottom layers
        h = hs[-1]
        h = self.mid.block_1(h, time_emb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, time_emb)

        # Upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), time_emb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h
