import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Any
from sklearn.preprocessing import StandardScaler


class PoissonGoalNN(nn.Module):
    def __init__(self, input_dim: int = 10, hidden_dim: int = 64):
        super(PoissonGoalNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 2),
            nn.Softplus()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + 1e-4


class NeuralNetworkModel:
    FEATURE_COLS = [
        "home_elo", "away_elo", "elo_diff", "home_factor",
        "home_sigma", "away_sigma",
        "home_avg_scored", "home_avg_conceded",
        "away_avg_scored", "away_avg_conceded"
    ]

    def __init__(self, weights_path: str = "models/faurecast_nn.pt", lr: float = 0.002):
        self.weights_path = weights_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PoissonGoalNN(input_dim=len(self.FEATURE_COLS)).to(self.device)
        self.scaler = StandardScaler()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.criterion = nn.PoissonNLLLoss(log_input=False)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 120, batch_size: int = 32) -> float:
        self.model.train()
        X_scaled = self.scaler.fit_transform(X)

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        final_loss = 0.0
        for epoch in range(epochs):
            for batch_x, batch_y in loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                final_loss = loss.item()

        self.save_weights()
        return final_loss

    def predict_xg(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            X_scaled = self.scaler.transform(X)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            lambdas = self.model(X_tensor).cpu().numpy()
        return lambdas

    def predict_match_details(self, X: np.ndarray, max_goals: int = 6) -> List[Dict[str, Any]]:
        lambdas = self.predict_xg(X)
        results = []

        for idx in range(len(lambdas)):
            l_h, l_a = float(lambdas[idx][0]), float(lambdas[idx][1])

            # Matriz de probabilidad Poisson
            p_matrix = np.zeros((max_goals + 1, max_goals + 1))
            for i in range(max_goals + 1):
                p_i = (l_h ** i) * math.exp(-l_h) / math.factorial(i)
                for j in range(max_goals + 1):
                    p_j = (l_a ** j) * math.exp(-l_a) / math.factorial(j)
                    p_matrix[i, j] = p_i * p_j

            p_matrix /= p_matrix.sum()

            # Probabilidades acumuladas por signo 1X2
            p_1 = float(np.sum(np.tril(p_matrix, -1)))
            p_x = float(np.sum(np.diag(p_matrix)))
            p_2 = float(np.sum(np.triu(p_matrix, 1)))

            probs = {"1": p_1, "X": p_x, "2": p_2}
            predicted_1x2 = max(probs, key=probs.get)

            # --- CORRECCIÓN: Filtrar la matriz según el signo 1X2 ganador ---
            mask = np.zeros_like(p_matrix, dtype=bool)
            if predicted_1x2 == "1":
                mask[np.tril_indices(max_goals + 1, -1)] = True  # Solo i > j
            elif predicted_1x2 == "2":
                mask[np.triu_indices(max_goals + 1, 1)] = True   # Solo i < j
            else:
                np.fill_diagonal(mask, True)                      # Solo i == j

            masked_p_matrix = np.where(mask, p_matrix, -1.0)
            best_i, best_j = np.unravel_index(np.argmax(masked_p_matrix), p_matrix.shape)

            results.append({
                "xg_home": l_h,
                "xg_away": l_a,
                "exact_home": int(best_i),
                "exact_away": int(best_j),
                "exact_score": f"{best_i}-{best_j}",
                "predicted_1x2": predicted_1x2,
                "confidence": float(probs[predicted_1x2]),
                "p_1": p_1, "p_x": p_x, "p_2": p_2
            })

        return results

    def save_weights(self) -> None:
        os.makedirs(os.path.dirname(self.weights_path), exist_ok=True)
        torch.save({"model_state": self.model.state_dict(), "scaler": self.scaler}, self.weights_path)

    def load_weights(self) -> bool:
        if os.path.exists(self.weights_path):
            checkpoint = torch.load(self.weights_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state"])
            self.scaler = checkpoint["scaler"]
            self.model.eval()
            return True
        return False