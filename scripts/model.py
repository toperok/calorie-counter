import torch
import torch.nn as nn
import timm
from transformers import AutoModel

class MultimodalModel(nn.Module):
    
    def __init__(
            self,
            config
    ) -> None:
        super().__init__()

        self.text_model = AutoModel.from_pretrained(config.TEXT_MODEL_NAME)
        self.image_model = timm.create_model(
            model_name=config.IMAGE_MODEL_NAME,
            pretrained=True,
            num_classes=0
        )

        self.text_proj = nn.Linear(self.text_model.config.hidden_size, config.HIDDEN_DIM)
        self.image_proj = nn.Linear(self.image_model.num_features, config.HIDDEN_DIM)
        self.mass_proj = nn.Linear(1, config.HIDDEN_DIM // 4)

        self.classifier = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM + 1, config.HIDDEN_DIM // 2),
            nn.LayerNorm(config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM // 2, config.OUTPUT_DIM)
        )
    
    def forward(
            self,
            input_ids,
            attention_mask,
            image,
            mass
    ):
        text_features = self.text_model(input_ids, attention_mask).last_hidden_state[:, 0, :]
        image_featurs = self.image_model(image)

        text_emb = self.text_proj(text_features)
        image_emb = self.image_proj(image_featurs)
        fused_emb = text_emb * image_emb

        final_input = torch.cat([fused_emb, mass.unsqueeze(1)], dim=1)

        return self.classifier(final_input)