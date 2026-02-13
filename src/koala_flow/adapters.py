# koala_flow.adapters - Universal model loading
# 
# This module provides a standard interface for loading different model types
# (scikit-learn, xgboost, lightgbm, pytorch, onnx, etc.) so the core pipeline doesn't care.

from abc import ABC, abstractmethod
from typing import Any, List, Optional
import os
import logging

logger = logging.getLogger("koala_flow.adapters")

class ModelAdapter(ABC):
    """
    Abstract Base Class for all model wrappers.
    """
    def __init__(self, path: str, version: Optional[str] = None):
        """
        Args:
            path: Path to the serialized model file (pkl, onnx, json).
            version: Optional version string for metadata tracking.
        """
        self.path = path
        self.version = version
        self.model = self.load(path)

    @abstractmethod
    def load(self, path: str) -> Any:
        """
        Implement the logic to load the model from disk.
        """
        pass

    @abstractmethod
    def predict(self, data: Any) -> List[Any]:
        """
        Run inference on the pre-processed data.
        Return a list or array of predictions.
        """
        pass


class PickleAdapter(ModelAdapter):
    """
    Generic adapter for any model saved with pickle (e.g. scikit-learn).
    """
    def load(self, path: str):
        import pickle
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        with open(path, "rb") as f:
            return pickle.load(f)

    def predict(self, data: Any):
        # Narwhals ensures data is pandas/polars compatible.
        # Sklearn models usually expect numpy or pandas.
        # We convert to native pandas/polars just in case.
        if hasattr(data, "to_pandas"):
            df = data.to_pandas()
        else:
            df = data
            
        # In a real library, handle predict_proba vs predict based on config
        return self.model.predict(df)


class XGBoostAdapter(ModelAdapter):
    """
    Specialized adapter for XGBoost models (json/ubj format).
    """
    def load(self, path: str):
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("xgboost is required for XGBoostAdapter. Install with 'pip install xgboost'.")
            
        if not os.path.exists(path):
             raise FileNotFoundError(f"Model file not found: {path}")

        bst = xgb.Booster()
        bst.load_model(path)
        return bst

    def predict(self, data: Any):
        import xgboost as xgb
        # XGBoost expects DMatrix
        # Convert Narwhals/Polars/Pandas to something XGBoost accepts
        if hasattr(data, "to_pandas"):
             data = data.to_pandas()
             
        dmatrix = xgb.DMatrix(data)
        return self.model.predict(dmatrix)


class PyTorchAdapter(ModelAdapter):
    """
    Adapter for PyTorch models (TorchScript or entire model pickle).
    """
    def load(self, path: str):
        try:
            import torch
        except ImportError:
             raise ImportError("torch is required for PyTorchAdapter. Install with 'pip install torch'.")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
            
        try:
            # Try loading as TorchScript first
            model = torch.jit.load(path)
        except Exception:
            logger.info("JIT load failed, trying standard torch.load")
            model = torch.load(path)
            
        model.eval()
        return model

    def predict(self, data: Any):
        import torch
        import numpy as np
        
        # Convert data to numpy float32, then tensor
        if hasattr(data, "to_numpy"):
            np_data = data.to_numpy()
        elif hasattr(data, "to_pandas"):
             np_data = data.to_pandas().to_numpy()
        else:
             np_data = np.array(data)
             
        tensor_data = torch.from_numpy(np_data.astype(np.float32))
        
        with torch.no_grad():
            output = self.model(tensor_data)
            
        # Return as list or numpy array for downstream consistency
        if hasattr(output, "numpy"):
            return output.numpy().tolist()
        return output


class ONNXAdapter(ModelAdapter):
    """
    Adapter for ONNX models using onnxruntime.
    """
    def load(self, path: str):
        try:
            import onnxruntime as ort
        except ImportError:
             raise ImportError("onnxruntime is required for ONNXAdapter. Install with 'pip install onnxruntime'.")
             
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        return ort.InferenceSession(path)

    def predict(self, data: Any):
        import numpy as np
        
        # Convert to numpy
        if hasattr(data, "to_numpy"):
            np_data = data.to_numpy()
        elif hasattr(data, "to_pandas"):
             np_data = data.to_pandas().to_numpy()
        else:
             np_data = np.array(data)
        
        np_data = np_data.astype(np.float32)

        # Get input name from the session
        input_name = self.model.get_inputs()[0].name
        label_name = self.model.get_outputs()[0].name
        
        # Run inference
        pred_onx = self.model.run([label_name], {input_name: np_data})[0]
        
        # Flatten/format as needed (often returns list of arrays)
        return pred_onx.tolist()
