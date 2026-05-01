from __future__ import annotations

import torch
import torch.nn as nn


class LSTMCoordinateModel(nn.Module):
    """BiLSTM coordinate regressor with AP identity and attention pooling."""

    def __init__(
        self,
        in_dim: int,
        coordinate_std: torch.Tensor | list[float] | tuple[float, float] | None = None,
    ) -> None:
        super().__init__()
        if in_dim % 2 != 0:
            raise ValueError(f"in_dim must be even, got {in_dim}")

        self.model_name = "lstm_coordinates"
        self.n_waps = in_dim // 2

        token_dim = 32
        ap_embed_dim = 16
        hidden_size = 384
        dropout = 0.25

        self.token_proj = nn.Sequential(
            nn.Linear(2, token_dim),
            nn.LayerNorm(token_dim),
            nn.ReLU(),
        )
        self.ap_embedding = nn.Embedding(self.n_waps, ap_embed_dim)

        self.lstm = nn.LSTM(
            input_size=token_dim + ap_embed_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )

        feat_dim = hidden_size * 2
        self.attn_pool = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

        self.loss_fn = nn.MSELoss()
        self.register_buffer("coordinate_std", None)
        if coordinate_std is not None:
            std = torch.as_tensor(coordinate_std, dtype=torch.float32)
            if std.shape != (2,):
                raise ValueError("coordinate_std must have shape (2,).")
            self.coordinate_std = std
            
    
    ###################################################################################################################
    # Critial Input Transformation 
    # We add attention lalyer to the model and get scores for every AP on every row/sample in the model,  
    # For every row/sample, the model learns how much each AP should contribute to the final prediction representation.
    ###################################################################################################################
    
    # References:
        
    # https://github.com/zhijing-jin/pytorch_RelationExtraction_AttentionBiLSTM/blob/master/model.py
        
    # https://www.youtube.com/watch?v=5g8wIumkZwg
    # Improving the LTSM performance - Bidirectional LSTM
    #                                - Attention Pooling
    # https://www.datacamp.com/de/tutorial/lstm-models
    # https://www.kaggle.com/code/robertke94/pytorch-bi-lstm-attention 
    

    def _to_sequence(self, x: torch.Tensor) -> torch.Tensor:
        
        """
        Convert the shared flat Wi-Fi fingerprint input into an identity-aware
        AP-level sequence for the LSTM.

        Original input layout:
            x shape = (batch, 2 * n_waps)

            x =
            [RSSI_1, RSSI_2, ..., RSSI_520,
             Mask_1, Mask_2, ..., Mask_520]

        Output layout:
            sequence shape = (batch, n_waps, token_dim + ap_embed_dim)

        Transformation:
            1. Split flat input into RSSI and detection-mask halves.
            2. Stack them into per-AP tokens: [RSSI_k, Mask_k].
            3. Project each 2D token into a learned token feature.
            4. Add a learned AP identity embedding.
            5. Concatenate token content and AP identity for LSTM input.

        Intuition:
            The LSTM should not see the input as one long flat vector. It should
            see a sequence of AP-level observations, where each step corresponds
            to one access point. The AP identity embedding is important because
            the same RSSI/mask values have different meaning depending on which
            AP produced them.
        """
        
        # Check that the input is a 2D tensor:
        #
        #   x shape should be (batch, 2 * n_waps)
        #
        # For UJIIndoorLoc:
        #
        #   n_waps = 520
        #   2 * n_waps = 1040
        #
        # The first half stores normalized RSSI values.
        # The second half stores detection-mask values.
        #
        # If the shape is wrong, the split below would silently produce invalid
        # RSSI/mask pairing, so we fail early.        
        if x.ndim != 2 or x.shape[1] != self.n_waps * 2:
            raise ValueError(
                f"Expected input shape (batch, {self.n_waps * 2}), got {tuple(x.shape)}"
            )
        
        # Extract the RSSI half of the flat vector.
        #
        # Input:
        #   x shape = (batch, 1040)
        #
        # Slice:
        #   x[:, : self.n_waps]
        #
        # Output:
        #   rssi shape = (batch, 520)
        #
        # Each column corresponds to one AP:
        #
        #   rssi[:, 0]   -> AP_1 RSSI
        #   rssi[:, 1]   -> AP_2 RSSI
        #   ...
        #   rssi[:, 519] -> AP_520 RSSI
        rssi = x[:, : self.n_waps]
        
        # Extract the detection-mask half of the flat vector.
        #
        # Slice:
        #   x[:, self.n_waps :]
        #
        # Output:
        #   mask shape = (batch, 520)
        #
        # Each value indicates whether the corresponding AP was detected:
        #
        #   mask[:, 0]   -> AP_1 detection indicator
        #   mask[:, 1]   -> AP_2 detection indicator
        #   ...
        #   mask[:, 519] -> AP_520 detection indicator
        #
        # This lets the model distinguish:
        #   - a weak observed signal
        #   - a truly missing AP reading
        mask = x[:, self.n_waps :]
        
        # Stack RSSI and mask together per AP.
        #
        # Before stacking:
        #
        #   rssi shape = (batch, 520)
        #   mask shape = (batch, 520)
        #
        # Operation:
        #
        #   seq = torch.stack([rssi, mask], dim=2)
        #
        # Output:
        #
        #   seq shape = (batch, 520, 2)
        #
        # Now each sequence position is one AP token:
        #
        #   seq[:, 0, :]   = [AP_1 RSSI, AP_1 mask]
        #   seq[:, 1, :]   = [AP_2 RSSI, AP_2 mask]
        #   ...
        #   seq[:, 519, :] = [AP_520 RSSI, AP_520 mask]
        #
        # ASCII view for one sample:
        #
        #   original flat input:
        #
        #   [rssi_1 rssi_2 ... rssi_520 | mask_1 mask_2 ... mask_520]
        #
        #   converted sequence:
        #
        #   [
        #     [rssi_1,   mask_1],
        #     [rssi_2,   mask_2],
        #     ...
        #     [rssi_520, mask_520]
        #   ]
        seq = torch.stack([rssi, mask], dim=2)

        # Project each raw 2D AP token into a learned token representation.
        #
        # Input:
        #   seq shape = (batch, 520, 2)
        #
        # self.token_proj is typically:
        #
        #   Linear(2 -> token_dim)
        #   LayerNorm(token_dim)
        #   ReLU()
        #
        # Output:
        #   token_feat shape = (batch, 520, token_dim)
        #
        # Conceptually:
        #
        #   [rssi_k, mask_k] -> learned token feature for AP_k
        #
        # This gives the LSTM a richer feature vector at each AP position
        # instead of feeding only two raw scalar values.
        token_feat = self.token_proj(seq)
        
        # Create integer AP identifiers:
        #
        #   ap_ids = [0, 1, 2, ..., 519]
        #
        # Shape:
        #   ap_ids shape = (520,)
        #
        # device=x.device is important because the IDs must live on the same
        # device as the input tensor, especially when training on GPU.
        ap_ids = torch.arange(self.n_waps, device=x.device)
        
        # Look up a learned embedding vector for each AP identity.
        #
        # self.ap_embedding(ap_ids):
        #   input shape  = (520,)
        #   output shape = (520, ap_embed_dim)
        #
        # unsqueeze(0):
        #   shape becomes (1, 520, ap_embed_dim)
        #
        # expand(x.shape[0], -1, -1):
        #   repeat the same AP identity table across the batch
        #
        # final shape:
        #   ap_feat shape = (batch, 520, ap_embed_dim)
        #
        # Conceptually:
        #
        #   AP_1   -> e_1
        #   AP_2   -> e_2
        #   ...
        #   AP_520 -> e_520
        #
        # Why this matters:
        #   Two APs may have similar RSSI/mask values, but they correspond to
        #   different physical signal sources. The identity embedding lets the
        #   model know which AP produced each observation.
        ap_feat = self.ap_embedding(ap_ids).unsqueeze(0).expand(x.shape[0], -1, -1)
        
        
        # Concatenate the learned token content and AP identity embedding.
        #
        # token_feat shape = (batch, 520, token_dim)
        # ap_feat shape    = (batch, 520, ap_embed_dim)
        #
        # Concatenation along dim=2 produces:
        #
        #   output shape = (batch, 520, token_dim + ap_embed_dim)
        #
        # Each AP sequence element becomes:
        #
        #   AP_k representation =
        #   [ token_proj(rssi_k, mask_k) | ap_embedding(k) ]
        #
        # ASCII view:
        #
        #   AP_1   -> [content_1   | identity_1]
        #   AP_2   -> [content_2   | identity_2]
        #   ...
        #   AP_520 -> [content_520 | identity_520]
        #
        # This final tensor is the identity-aware AP sequence consumed by
        # the Bidirectional LSTM.
        return torch.cat([token_feat, ap_feat], dim=2)

   
    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        
        """
        Encode the shared flat Wi-Fi fingerprint input into one fixed-size
        representation for prediction.

        This function does three things:

        1. Converts the flat input vector into an AP-level sequence.
           The helper `_to_sequence()` changes the input from:

               (batch, 1040)

           into:

               (batch, 520, token_dim + ap_embed_dim)

           where each sequence element represents one access point with both
           signal content and AP identity information.

        2. Runs the AP sequence through the bidirectional LSTM.
           The BiLSTM produces one contextual hidden state for each AP.

        3. Applies learned attention pooling.
           Instead of averaging all AP hidden states equally, the model learns
           which AP positions are more informative and computes a weighted sum.

        Final output:
            pooled shape = (batch, 2 * hidden_size)

        This pooled vector is then passed to the classifier or regression head.
        """
        
        seq = self._to_sequence(x)  #  Call the _to_sequence function above                
        
        # Run the AP sequence through the bidirectional LSTM.
        #
        # Input:
        #   seq shape = (batch, n_waps, token_dim + ap_embed_dim)
        #
        # In our case:
        #   n_waps = 520
        #
        # Each sequence position corresponds to one access point:
        #
        #   seq[:, 0, :]   -> AP_1 representation
        #   seq[:, 1, :]   -> AP_2 representation
        #   ...
        #   seq[:, 519, :] -> AP_520 representation
        #
        # Because the LSTM is bidirectional, each AP output vector contains
        # information from both directions of the AP sequence:
        #
        #   forward direction:  AP_1 -> AP_2 -> ... -> AP_520
        #   backward direction: AP_520 -> AP_519 -> ... -> AP_1
        #
        # Output:
        #   out shape = (batch, n_waps, 2 * hidden_size)
        #
        # For each AP_i, out[:, i, :] is the contextual representation h_i.
        #
        # The second returned value is (h_n, c_n), the final hidden and cell states.
        # We do not use it here because we want all AP-level hidden states, not just
        # the final LSTM state.
        out, _ = self.lstm(seq)
        
        
        # Compute one attention score for each AP hidden state.
        #
        # Input:
        #   out shape = (batch, n_waps, 2 * hidden_size)
        #
        # self.attn_pool is a small neural network:
        #
        #   Linear(2 * hidden_size -> 128)
        #   Tanh()
        #   Linear(128 -> 1)
        #
        # It is applied independently to each AP hidden state h_i.
        #
        # Conceptually:
        #
        #   h_1 -> score_1
        #   h_2 -> score_2
        #   h_3 -> score_3
        #   ...
        #   h_520 -> score_520
        #
        # Before squeeze:
        #   self.attn_pool(out) shape = (batch, n_waps, 1)
        #
        # After squeeze(-1):
        #   attn_logits shape = (batch, n_waps)
        #
        # These values are raw, unnormalized attention scores. They can be positive
        # or negative, and they are not probabilities yet.
        attn_logits = self.attn_pool(out).squeeze(-1)
        
        
        # Convert raw attention scores into normalized attention weights.
        #
        # Input:
        #   attn_logits shape = (batch, n_waps)
        #
        # softmax is applied along dim=1, the AP-sequence dimension.
        # This means each sample gets one probability distribution over APs.
        #
        # For one sample:
        #
        #   attn_logits = [s_1, s_2, ..., s_520]
        #
        # softmax converts this into:
        #
        #   attn_weights = [a_1, a_2, ..., a_520]
        #
        # where:
        #
        #   a_i = exp(s_i) / sum_j exp(s_j)
        #
        # and:
        #
        #   sum_i a_i = 1
        #   a_i >= 0
        #
        # Interpretation:
        #   a_i measures how much AP_i contributes to the final pooled representation.
        #
        # Before unsqueeze:
        #   torch.softmax(attn_logits, dim=1) shape = (batch, n_waps)
        #
        # After unsqueeze(-1):
        #   attn_weights shape = (batch, n_waps, 1)
        #
        # The final singleton dimension is added so that PyTorch can broadcast the
        # weights over the hidden dimension of out during multiplication.
        attn_weights = torch.softmax(attn_logits, dim=1).unsqueeze(-1)
        
        
        # Compute the final fixed-size representation by weighted-sum pooling.
        #
        # Shapes:
        #   attn_weights shape = (batch, n_waps, 1)
        #   out shape          = (batch, n_waps, 2 * hidden_size)
        #
        # Broadcasting:
        #   attn_weights is broadcast across the hidden dimension, so each AP hidden
        #   vector h_i is multiplied by its scalar attention weight a_i.
        #
        # Elementwise:
        #
        #   attn_weights * out
        #
        # produces:
        #
        #   [a_1 * h_1,
        #    a_2 * h_2,
        #    ...
        #    a_520 * h_520]
        #
        # Then we sum over dim=1, the AP dimension:
        #
        #   pooled = sum_i a_i * h_i
        #
        # Output:
        #   pooled shape = (batch, 2 * hidden_size)
        #
        # This gives the downstream classifier one vector per sample.
        #
        # Why this is useful:
        #   Mean pooling would treat every AP hidden state equally.
        #   Final-state pooling would rely only on the last recurrent state.
        #   Attention pooling lets the model learn which APs are more informative
        #   for the current prediction and weight them more strongly.
        #
        # Important distinction:
        #   This is learned attention pooling, not Transformer self-attention.
        #   It does not compute pairwise AP-to-AP attention. The BiLSTM creates
        #   contextual AP representations first, and this layer only learns how to
        #   aggregate those representations into one final vector.
        pooled = torch.sum(attn_weights * out, dim=1)
        return pooled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self._encode(x))

    def compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(outputs, targets.to(dtype=outputs.dtype))

    def evaluate_outputs(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> dict[str, float]:
        if self.coordinate_std is None:
            raise ValueError("Coordinate evaluation in meters requires coordinate_std.")

        pred = outputs.to(dtype=torch.float64)
        true = targets.to(dtype=torch.float64)
        diff = pred - true

        euclidean_norm = torch.linalg.norm(diff, dim=1)
        coord_euclidean_norm = float(torch.mean(euclidean_norm).item())
        coord_rmse_norm = float(torch.sqrt(torch.mean(euclidean_norm.square())).item())

        std = self.coordinate_std.to(dtype=torch.float64)
        lon_diff_m = diff[:, 0] * std[0]
        lat_diff_m = diff[:, 1] * std[1]
        euclidean_m = torch.sqrt(lon_diff_m.square() + lat_diff_m.square())
        coord_euclidean_m = float(torch.mean(euclidean_m).item())
        coord_rmse_m = float(torch.sqrt(torch.mean(euclidean_m.square())).item())

        return {
            "score": -coord_euclidean_m,
            "coordinate_mean_euclidean": coord_euclidean_norm,
            "coordinate_rmse": coord_rmse_norm,
            "coordinate_mean_euclidean_m": coord_euclidean_m,
            "coordinate_rmse_m": coord_rmse_m,
        }
