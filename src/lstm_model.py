"""
LSTM Encoder + Trainer.

Architecture :
    Input (batch, 91, 1)
    → LSTM multicouche
    → Dernier hidden state
    → Linear → ReLU → BatchNorm → embedding (batch, 32)
    → Dropout → Linear(1)  [classifier head — entraînement seulement]

Points critiques :
  - WeightedRandomSampler : sur-échantillonne les positifs (déséquilibre)
  - BCEWithLogitsLoss + pos_weight complémentaire
  - Early stopping sur val_AUC (PAS val_loss)
  - Gradient clipping max_norm=1.0
  - ReduceLROnPlateau (mode='max') sur val_AUC
  - En extraction : model.encode() PAS model.forward()
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
import logging

logger = logging.getLogger(__name__)


class LSTMEncoder(nn.Module):

    def __init__(self, input_size=1, hidden_size=64, num_layers=2,
                 dropout=0.2, bidirectional=False, embedding_dim=32):
        super().__init__()
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        lstm_out = hidden_size * self.num_directions
        self.projection = nn.Sequential(
            nn.Linear(lstm_out, embedding_dim),
            nn.ReLU(),
            nn.BatchNorm1d(embedding_dim),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(embedding_dim, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[n // 4: n // 2].fill_(1.0)  # forget gate bias = 1
        for module in self.projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                module.bias.data.fill_(0)

    def encode(self, x):
        """
        Extraction d'embedding uniquement (inférence + extraction post-entraînement).
        Doit être appelé avec model.eval() + torch.no_grad().
        """
        _, (h_n, _) = self.lstm(x)
        if self.bidirectional:
            hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            hidden = h_n[-1]
        return self.projection(hidden)

    def forward(self, x):
        """Forward complet avec tête de classification (entraînement uniquement)."""
        emb = self.encode(x)
        logits = self.classifier(emb).squeeze(-1)
        return logits, emb


class LSTMTrainer:

    def __init__(self, config: dict, save_dir: str = "models"):
        self.config = config
        self.save_dir = save_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        os.makedirs(save_dir, exist_ok=True)
        logger.info(f"LSTMTrainer device : {self.device}")

    def train(
        self,
        sequences: torch.Tensor,
        targets: np.ndarray,
        val_sequences: torch.Tensor = None,
        val_targets: np.ndarray = None,
    ):
        """
        Entraîne le LSTM encoder.
        Returns:
            model   : LSTMEncoder (meilleur checkpoint selon val_AUC)
            history : dict(train_loss, val_loss, val_auc, lr)
        """
        cfg = self.config

        # ── Auto-split si validation non fournie ──
        if val_sequences is None:
            idx_tr, idx_val = train_test_split(
                np.arange(len(targets)), test_size=0.2,
                stratify=targets, random_state=42,
            )
            val_sequences = sequences[idx_val]
            val_targets   = targets[idx_val]
            sequences     = sequences[idx_tr]
            targets       = targets[idx_tr]

        # ── WeightedRandomSampler ──
        n_pos = int((targets == 1).sum())
        n_neg = int((targets == 0).sum())
        if n_pos == 0:
            raise ValueError("Aucun positif dans les données d'entraînement.")
        class_weights = {0: 1.0 / n_neg, 1: 1.0 / n_pos}
        sample_weights = torch.tensor([class_weights[int(t)] for t in targets])
        sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(targets), replacement=True
        )

        train_ds = TensorDataset(sequences, torch.FloatTensor(targets))
        val_ds   = TensorDataset(val_sequences, torch.FloatTensor(val_targets))

        train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], sampler=sampler)
        val_loader   = DataLoader(val_ds,   batch_size=cfg['batch_size'], shuffle=False)

        # ── Modèle ──
        model = LSTMEncoder(
            input_size    = cfg['input_size'],
            hidden_size   = cfg['hidden_size'],
            num_layers    = cfg['num_layers'],
            dropout       = cfg['dropout'],
            bidirectional = cfg['bidirectional'],
            embedding_dim = cfg['embedding_dim'],
        ).to(self.device)

        # ── Loss + Optimizer + Scheduler ──
        pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(self.device)
        criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer  = torch.optim.AdamW(
            model.parameters(), lr=cfg['learning_rate'], weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
        )

        history = {'train_loss': [], 'val_loss': [], 'val_auc': [], 'lr': []}
        best_auc      = 0.0
        patience_ctr  = 0
        best_path     = os.path.join(self.save_dir, "lstm_encoder.pt")

        print(f"\n   {'Epoch':>6} {'TrainLoss':>10} {'ValLoss':>10} {'ValAUC':>8} {'LR':>10}  Status")
        print(f"   {'─'*6} {'─'*10} {'─'*10} {'─'*8} {'─'*10}  {'─'*10}")

        for epoch in range(cfg['epochs']):
            # ── Train ──
            model.train()
            train_loss = 0.0
            for bx, by in train_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                logits, _ = model(bx)
                loss = criterion(logits, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item() * len(bx)
            train_loss /= len(train_loader.dataset)

            # ── Validation ──
            model.eval()
            val_loss  = 0.0
            val_preds = []
            val_true  = []
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(self.device), by.to(self.device)
                    logits, _ = model(bx)
                    val_loss += criterion(logits, by).item() * len(bx)
                    val_preds.extend(torch.sigmoid(logits).cpu().numpy())
                    val_true.extend(by.cpu().numpy())
            val_loss /= len(val_loader.dataset)
            val_auc   = roc_auc_score(np.array(val_true), np.array(val_preds))
            current_lr = optimizer.param_groups[0]['lr']

            scheduler.step(val_auc)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_auc'].append(val_auc)
            history['lr'].append(current_lr)

            # ── Early stopping ──
            status = ""
            if val_auc > best_auc:
                best_auc     = val_auc
                patience_ctr = 0
                torch.save(model.state_dict(), best_path)
                status = "★ best"
            else:
                patience_ctr += 1

            if (epoch + 1) % 5 == 0 or epoch == 0 or status:
                print(
                    f"   {epoch+1:6d} {train_loss:10.5f} {val_loss:10.5f} "
                    f"{val_auc:8.4f} {current_lr:10.2e}  {status}"
                )

            if patience_ctr >= cfg['patience']:
                print(f"   Early stopping à l'epoch {epoch+1}")
                break

        # Charger le meilleur checkpoint
        model.load_state_dict(torch.load(best_path, weights_only=True))
        model.to(self.device)
        print(f"\n   ✅ Meilleur val_AUC : {best_auc:.4f}")
        return model, history

    def extract_embeddings(
        self,
        model: LSTMEncoder,
        sequences: torch.Tensor,
        batch_size: int = 512,
    ) -> np.ndarray:
        """
        Extrait les embeddings via model.encode() (PAS forward).
        Requiert model.eval() + torch.no_grad().
        Returns: np.ndarray (n, embedding_dim)
        """
        model.eval()
        all_emb = []
        loader = DataLoader(
            TensorDataset(sequences), batch_size=batch_size, shuffle=False
        )
        with torch.no_grad():
            for (bx,) in loader:
                bx  = bx.to(self.device)
                emb = model.encode(bx)
                all_emb.append(emb.cpu().numpy())
        return np.concatenate(all_emb, axis=0)

    def cross_validate(
        self,
        sequences: torch.Tensor,
        targets: np.ndarray,
        n_folds: int = 5,
    ) -> dict:
        """
        Stratified K-Fold CV pour évaluer la stabilité du LSTM.
        Returns: dict(fold_aucs, mean_auc, std_auc)
        """
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_aucs = []

        for fold, (tr_idx, val_idx) in enumerate(
            skf.split(np.zeros(len(targets)), targets)
        ):
            print(f"\n   ── Fold {fold+1}/{n_folds} ──")
            _, hist = self.train(
                sequences[tr_idx], targets[tr_idx],
                sequences[val_idx], targets[val_idx],
            )
            fold_aucs.append(max(hist['val_auc']))

        mean_auc = float(np.mean(fold_aucs))
        std_auc  = float(np.std(fold_aucs))
        print(f"\n   Cross-validation LSTM ({n_folds} folds):")
        print(f"   AUC par fold : {[f'{a:.4f}' for a in fold_aucs]}")
        print(f"   Moyenne ± std : {mean_auc:.4f} ± {std_auc:.4f}")

        if mean_auc < 0.55:
            print("   ⚠️  AUC moyen < 0.55 : le LSTM n'apprend pas — vérifier les données.")
        elif mean_auc < 0.65:
            print("   ℹ️  AUC 0.55-0.65 : normal pour un encodeur, embeddings complémentaires.")
        else:
            print("   ✅ AUC > 0.65 : LSTM capture un fort signal.")

        return {'fold_aucs': fold_aucs, 'mean_auc': mean_auc, 'std_auc': std_auc}
