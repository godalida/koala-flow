# koala_flow - Lightweight Batch Inference Framework
# 
# The core library is structured around 3 main pillars:
# 1. Source (dlt) - Fetches data.
# 2. Transform + Predict (Narwhals + Model) - Preps features and runs inference.
# 3. Sink (dlt) - Saves results.

import narwhals as nw
import dlt
from typing import Any, Callable, Optional, Union, Dict, List
import logging
from .adapters import ModelAdapter

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("koala_flow")


class InferencePipeline:
    """
    The main orchestrator. It connects a source (dlt resource), a model adapter,
    and a destination (dlt pipeline destination).
    """

    def __init__(
        self,
        name: str,
        model: ModelAdapter,
        destination: Any,  # String (e.g. 'duckdb') or dlt destination object
        dataset_name: str = "ml_predictions",
        prep_fn: Optional[Callable[[nw.DataFrame], nw.DataFrame]] = None,
    ):
        """
        Args:
            name: Name of the pipeline (for dlt metadata).
            model: An instance of a koala_flow.ModelAdapter wrapping your model.
            destination: Where the predictions go (e.g., 'duckdb', 'bigquery', 'snowflake').
            dataset_name: The dataset/schema name in the destination.
            prep_fn: A function that takes a Narwhals DataFrame and returns a transformed DataFrame ready for prediction.
        """
        self.name = name
        self.model = model
        self.destination = destination
        self.dataset_name = dataset_name
        self.prep_fn = prep_fn
        
        # Initialize the internal dlt pipeline
        self.dlt_pipeline = dlt.pipeline(
            pipeline_name=self.name,
            destination=self.destination,
            dataset_name=self.dataset_name
        )

    def _process_batch(self, batch: Any) -> List[Dict[str, Any]]:
        """
        Internal method to process a single batch of data from dlt.
        1. Convert to Narwhals DataFrame.
        2. Run user's prep_fn (feature engineering).
        3. Run model inference.
        4. Append metadata and return as list of dicts for dlt.
        """
        # Convert raw batch (list of dicts usually) to Narwhals DataFrame via Pandas/Polars
        # Narwhals handles the backend abstraction automatically if installed.
        # Here we assume standard dicts -> pandas/polars conversion first.
        # For simplicity in this MVP, we use narwhals.from_native() if it's already a DF,
        # or convert list-of-dicts to a dataframe first.
        
        try:
            # Simple list-of-dicts ingestion strategy (MVP)
            import pandas as pd
            native_df = pd.DataFrame(batch)
            df = nw.from_native(native_df)
        except Exception as e:
            logger.error(f"Failed to convert batch to DataFrame: {e}")
            raise

        # 1. Feature Engineering (User defined)
        if self.prep_fn:
            features_df = self.prep_fn(df)
        else:
            features_df = df

        # 2. Inference
        # The adapter handles the specifics of .predict(), .predict_proba(), etc.
        predictions = self.model.predict(features_df)

        # 3. Combine Results
        # We attach predictions back to the original records (or features)
        # Convert predictions to a Series/Column and add to DF
        
        # Note: In a real library, handle multiple output columns (prob_class_0, prob_class_1)
        results_df = features_df.with_columns(
            nw.new_series(
                name="prediction", 
                values=predictions, 
                native_namespace=nw.get_native_namespace(features_df)
            ),
            nw.lit(self.name).alias("pipeline_name"),
            nw.lit(self.model.version or "unknown").alias("model_version"),
            nw.lit(str(dlt.current_pipeline().pipeline_name)).alias("run_id") 
        )

        # 4. Convert back to Python dicts for dlt to load
        # dlt expects an iterable of dicts
        return results_df.to_native().to_dict(orient="records")

    def run(self, source: Any, table_name: str = "predictions"):
        """
        Executes the pipeline.
        
        Args:
            source: A dlt source or resource (iterator).
            table_name: The table name in the destination.
        """
        logger.info(f"Starting pipeline '{self.name}'...")

        # We create a generator that yields processed batches
        # This allows dlt to handle the extraction and loading in chunks
        def resource_wrapper():
            # If source is a dlt resource, we can iterate it
            # In a full implementation, we'd handle dlt's specialized resource types better
            for batch in source:
                yield self._process_batch(batch)

        # Run the dlt pipeline
        # We wrap our logic in a dlt resource
        @dlt.resource(table_name=table_name)
        def processing_resource():
            # If the source is already a dlt resource, we might need to iterate its pages/batches
            # For MVP, assume source is an iterable of data (list of dicts or generator)
            
            # Simple chunking logic if source is a large list
            chunk_size = 1000
            buffer = []
            
            for record in source:
                buffer.append(record)
                if len(buffer) >= chunk_size:
                    logger.info(f"Processing batch of {len(buffer)} records...")
                    yield from self._process_batch(buffer)
                    buffer = []
            
            if buffer:
                logger.info(f"Processing final batch of {len(buffer)} records...")
                yield from self._process_batch(buffer)

        info = self.dlt_pipeline.run(processing_resource())
        logger.info(f"Pipeline finished. Load info: {info}")
        return info
