import torch.nn as nn
import torch

class AutoencoderSimple(nn.Module):
    def __init__(self, in_dim, out_dim, latent_dim, n_layers_hidden=2, activation="relu", needs_flatten=False):
        super().__init__()

        act_fn = nn.ELU if activation.lower() == "elu" else nn.ReLU
        
        hidden_encoder = []
        for _ in range(n_layers_hidden):
            hidden_encoder.extend([nn.Linear(512, 512), act_fn()])
            
        self.encoder = nn.Sequential(
            nn.Flatten(1) if needs_flatten else nn.Identity(),
            nn.Linear(in_dim, 512), 
            act_fn(),
            *hidden_encoder,
            nn.Linear(512, latent_dim)
        )
        
        hidden_decoder = []
        for _ in range(n_layers_hidden):
            hidden_decoder.extend([nn.Linear(512, 512), act_fn()])

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512), 
            act_fn(),
            *hidden_decoder,
            nn.Linear(512, out_dim),
            nn.Unflatten(1, (1, 28, 28)) if needs_flatten else nn.Identity()
            
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

