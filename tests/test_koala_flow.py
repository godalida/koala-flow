import pytest
import os
import sys
import pickle
from unittest.mock import MagicMock, patch
import pandas as pd
import narwhals as nw

# Ensure we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from koala_flow.core import InferencePipeline
from koala_flow.adapters import PickleAdapter, ModelAdapter, PyTorchAdapter, ONNXAdapter

# --- Mocks ---

class MockModel:
    def predict(self, data):
        # Return dummy predictions (0s and 1s)
        return [0] * len(data)

@pytest.fixture
def mock_pickle_model(tmp_path):
    model_path = tmp_path / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(MockModel(), f)
    return str(model_path)

@pytest.fixture
def mock_dlt_source():
    return [{"a": 1, "b": 2}, {"a": 3, "b": 4}]

# --- Tests ---

def test_pipeline_initialization():
    adapter = MagicMock(spec=ModelAdapter)
    pipeline = InferencePipeline(
        name="test_pipe",
        model=adapter,
        destination="duckdb"
    )
    assert pipeline.name == "test_pipe"

def test_pipeline_execution_mock(mock_pickle_model, mock_dlt_source):
    """Test full execution flow with a pickle adapter."""
    adapter = PickleAdapter(mock_pickle_model)
    
    # Mock dlt pipeline run to avoid actual DB writes
    with patch("dlt.pipeline") as mock_dlt:
        mock_instance = mock_dlt.return_value
        mock_instance.run.return_value = "Success"
        
        pipeline = InferencePipeline(
            name="test_run",
            model=adapter,
            destination="duckdb"
        )
        
        result = pipeline.run(mock_dlt_source, table_name="results")
        
        assert result == "Success"
        # Ensure run was called
        mock_instance.run.assert_called_once()

def test_pytorch_adapter_structure():
    """Test PyTorch adapter structure without needing torch installed (mocking imports)."""
    import numpy as np
    mock_torch = MagicMock()
    # Mock torch.load/jit.load
    mock_torch.load.return_value = MagicMock() # The model
    mock_torch.from_numpy.return_value = MagicMock()
    
    with patch.dict(sys.modules, {'torch': mock_torch}):
        # We also need os.path.exists to be True
        with patch("os.path.exists", return_value=True):
             adapter = PyTorchAdapter("fake_model.pt")
             assert adapter.model is not None
             
             # Test predict call
             data = pd.DataFrame({"a": [1, 2]})
             adapter.predict(data)
             
             # Verify torch was called
             mock_torch.from_numpy.assert_called()

def test_onnx_adapter_structure():
    """Test ONNX adapter structure without needing onnxruntime."""
    import numpy as np
    mock_ort = MagicMock()
    mock_sess = MagicMock()
    mock_ort.InferenceSession.return_value = mock_sess
    
    # Mock input/output metadata for session
    input_meta = MagicMock(); input_meta.name = "input"
    output_meta = MagicMock(); output_meta.name = "output"
    mock_sess.get_inputs.return_value = [input_meta]
    mock_sess.get_outputs.return_value = [output_meta]
    mock_sess.run.return_value = [MagicMock()] # returns list of results
    
    with patch.dict(sys.modules, {'onnxruntime': mock_ort}):
        with patch("os.path.exists", return_value=True):
            adapter = ONNXAdapter("fake_model.onnx")
            
            data = pd.DataFrame({"a": [1.0, 2.0]})
            adapter.predict(data)
            
            mock_sess.run.assert_called()

