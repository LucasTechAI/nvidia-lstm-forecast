"""
Application dependencies and shared state.
"""

import logging
from typing import Optional
from pathlib import Path

import torch
from sklearn.preprocessing import MinMaxScaler

from src.models.lstm_model import NvidiaLSTM, load_model_from_checkpoint
from src.config import settings

logger = logging.getLogger(__name__)


class ModelState:
    """Singleton to hold loaded model and scaler state."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.model: Optional[NvidiaLSTM] = None
        self.scaler: Optional[MinMaxScaler] = None
        self.model_config: dict = {}
        self.device: str = "cpu"
        self.is_training: bool = False
        self.training_run_id: Optional[str] = None
        self.current_epoch: int = 0
        self.total_epochs: int = 0
        self.current_loss: float = 0.0
        
        self._initialized = True
    
    def load_model(
        self, 
        checkpoint_path: Optional[str] = None,
        scaler_path: Optional[str] = None
    ) -> bool:
        """Load model and scaler from checkpoint."""
        try:
            self.device = settings.get_device()
            
            # Default paths
            if checkpoint_path is None:
                checkpoint_path = str(settings.models_dir / "checkpoints" / "best_model.pt")
            
            if scaler_path is None:
                # Try to find scaler in outputs
                scaler_dir = settings.outputs_dir / "artifacts"
                if scaler_dir.exists():
                    for run_dir in scaler_dir.iterdir():
                        scaler_file = run_dir / "scaler.joblib"
                        if scaler_file.exists():
                            scaler_path = str(scaler_file)
                            break
            
            # Load model
            if Path(checkpoint_path).exists():
                checkpoint = torch.load(
                    checkpoint_path, 
                    map_location=self.device,
                    weights_only=False
                )
                self.model_config = checkpoint.get('model_config', {})
                
                self.model = NvidiaLSTM(
                    input_size=self.model_config.get('input_size', 1),
                    hidden_size=self.model_config.get('hidden_size', 128),
                    num_layers=self.model_config.get('num_layers', 2),
                    output_size=self.model_config.get('output_size', 1),
                    dropout=self.model_config.get('dropout', 0.2),
                    bidirectional=self.model_config.get('bidirectional', False)
                )
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model = self.model.to(self.device)
                self.model.eval()
                
                logger.info(f"Model loaded from {checkpoint_path}")
            else:
                logger.warning(f"Checkpoint not found: {checkpoint_path}")
                return False
            
            # Load scaler
            if scaler_path and Path(scaler_path).exists():
                import joblib
                self.scaler = joblib.load(scaler_path)
                logger.info(f"Scaler loaded from {scaler_path}")
            else:
                logger.warning("Scaler not found, predictions may not work correctly")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    @property
    def is_ready(self) -> bool:
        """Check if model is ready for inference."""
        return self.model is not None and self.scaler is not None


# Global model state
model_state = ModelState()


def get_model_state() -> ModelState:
    """Dependency to get model state."""
    return model_state
