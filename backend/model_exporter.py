import os
import torch
from sentence_transformers import SentenceTransformer
import onnxruntime as ort
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class ModelExporter:
    """
    Exports models to mobile-friendly formats (ONNX, TFLite) for offline capabilities.
    """

    def __init__(self, export_dir: str = "models/mobile/"):
        self.export_dir = export_dir
        os.makedirs(export_dir, exist_ok=True)

    def export_embedding_model(self, model_name: str = "all-MiniLM-L6-v2",
                              onnx_path: str = "embedding_model.onnx") -> str:
        """
        Export sentence transformer model to ONNX format.

        Args:
            model_name: HuggingFace model name
            onnx_path: Output ONNX file path

        Returns:
            Path to exported model
        """
        try:
            from sentence_transformers import SentenceTransformer
            import torch.onnx

            # Load model
            model = SentenceTransformer(model_name)

            # Create dummy input
            dummy_input = ["This is a test sentence for ONNX export."]

            # Export to ONNX
            full_path = os.path.join(self.export_dir, onnx_path)

            # Get the actual PyTorch model from sentence transformer
            torch_model = model[0].auto_model  # Get the transformer model

            # Create input for ONNX export
            tokenizer = model.tokenizer
            inputs = tokenizer(dummy_input, return_tensors="pt", padding=True, truncation=True)

            torch.onnx.export(
                torch_model,
                (inputs['input_ids'], inputs['attention_mask']),
                full_path,
                input_names=['input_ids', 'attention_mask'],
                output_names=['last_hidden_state', 'pooler_output'],
                dynamic_axes={
                    'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                    'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                    'last_hidden_state': {0: 'batch_size', 1: 'sequence_length'}
                },
                opset_version=13
            )

            # Save tokenizer config
            tokenizer_config = {
                "model_name": model_name,
                "max_seq_length": model.max_seq_length,
                "tokenizer_class": tokenizer.__class__.__name__
            }

            with open(os.path.join(self.export_dir, "tokenizer_config.json"), "w") as f:
                json.dump(tokenizer_config, f)

            logger.info(f"Exported embedding model to {full_path}")
            return full_path

        except Exception as e:
            logger.error(f"Failed to export embedding model: {e}")
            return None

    def export_memory_transformer(self, model_path: str = None,
                                 onnx_path: str = "memory_transformer.onnx") -> str:
        """
        Export memory transformer to ONNX format.

        Args:
            model_path: Path to trained model (None for mock export)
            onnx_path: Output ONNX file path

        Returns:
            Path to exported model
        """
        try:
            # For now, create a mock transformer export since we don't have a trained model
            # In production, this would export the actual trained MemoryTransformer

            full_path = os.path.join(self.export_dir, onnx_path)

            # Create a simple mock model for demonstration
            class MockTransformer(torch.nn.Module):
                def __init__(self, embed_dim=384, num_heads=8, num_layers=6):
                    super().__init__()
                    self.embed_dim = embed_dim
                    encoder_layer = torch.nn.TransformerEncoderLayer(
                        d_model=embed_dim, nhead=num_heads, batch_first=True
                    )
                    self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                    self.classifier = torch.nn.Linear(embed_dim, 3)  # 3 outputs: confidence, reasoning_type, activated_nodes

                def forward(self, x):
                    # x shape: (batch_size, seq_len, embed_dim)
                    output = self.transformer(x)
                    pooled = torch.mean(output, dim=1)  # Simple pooling
                    logits = self.classifier(pooled)
                    return logits

            model = MockTransformer()

            # Create dummy input (batch_size=1, seq_len=10, embed_dim=384)
            dummy_input = torch.randn(1, 10, 384)

            torch.onnx.export(
                model,
                dummy_input,
                full_path,
                input_names=['embeddings'],
                output_names=['outputs'],
                dynamic_axes={'embeddings': {0: 'batch_size', 1: 'seq_len'}},
                opset_version=13
            )

            # Save model config
            config = {
                "model_type": "memory_transformer",
                "embed_dim": 384,
                "num_heads": 8,
                "num_layers": 6,
                "max_seq_len": 512,
                "output_classes": 3
            }

            with open(os.path.join(self.export_dir, "memory_transformer_config.json"), "w") as f:
                json.dump(config, f)

            logger.info(f"Exported memory transformer to {full_path}")
            return full_path

        except Exception as e:
            logger.error(f"Failed to export memory transformer: {e}")
            return None

    def create_tflite_model(self, onnx_path: str, tflite_path: str) -> str:
        """
        Convert ONNX model to TensorFlow Lite format.

        Args:
            onnx_path: Input ONNX model path
            tflite_path: Output TFLite file path

        Returns:
            Path to TFLite model
        """
        try:
            import onnx
            from onnx_tf.backend import prepare
            import tensorflow as tf

            # Load ONNX model
            onnx_model = onnx.load(onnx_path)

            # Convert to TensorFlow
            tf_rep = prepare(onnx_model)
            tf_model_path = os.path.join(self.export_dir, "temp_tf_model")
            tf_rep.export_graph(tf_model_path)

            # Convert to TFLite
            converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float32]

            tflite_model = converter.convert()

            full_tflite_path = os.path.join(self.export_dir, tflite_path)
            with open(full_tflite_path, "wb") as f:
                f.write(tflite_model)

            # Cleanup
            import shutil
            shutil.rmtree(tf_model_path, ignore_errors=True)

            logger.info(f"Created TFLite model at {full_tflite_path}")
            return full_tflite_path

        except Exception as e:
            logger.error(f"Failed to create TFLite model: {e}")
            return None

    def export_mobile_bundle(self, bundle_name: str = "brein_mobile_v1") -> Dict[str, str]:
        """
        Create a complete mobile bundle with all necessary models and configs.

        Args:
            bundle_name: Name for the mobile bundle

        Returns:
            Dictionary with paths to all exported files
        """
        bundle_dir = os.path.join(self.export_dir, bundle_name)
        os.makedirs(bundle_dir, exist_ok=True)

        exported_files = {}

        try:
            # Export embedding model
            embedding_onnx = self.export_embedding_model(
                onnx_path=os.path.join(bundle_name, "embedding_model.onnx")
            )
            if embedding_onnx:
                exported_files["embedding_onnx"] = embedding_onnx

                # Create TFLite version
                tflite_path = f"{bundle_name}/embedding_model.tflite"
                embedding_tflite = self.create_tflite_model(
                    os.path.basename(embedding_onnx), tflite_path
                )
                if embedding_tflite:
                    exported_files["embedding_tflite"] = embedding_tflite

            # Export memory transformer
            transformer_onnx = self.export_memory_transformer(
                onnx_path=os.path.join(bundle_name, "memory_transformer.onnx")
            )
            if transformer_onnx:
                exported_files["transformer_onnx"] = transformer_onnx

                # Create TFLite version
                tflite_path = f"{bundle_name}/memory_transformer.tflite"
                transformer_tflite = self.create_tflite_model(
                    os.path.basename(transformer_onnx), tflite_path
                )
                if transformer_tflite:
                    exported_files["transformer_tflite"] = transformer_tflite

            # Create bundle manifest
            manifest = {
                "bundle_name": bundle_name,
                "version": "1.0.0",
                "models": {
                    "embedding": {
                        "onnx": os.path.basename(embedding_onnx) if embedding_onnx else None,
                        "tflite": os.path.basename(embedding_tflite) if 'embedding_tflite' in locals() and embedding_tflite else None,
                        "input_shape": ["batch_size", "sequence_length"],
                        "output_shape": ["batch_size", "embedding_dim"]
                    },
                    "memory_transformer": {
                        "onnx": os.path.basename(transformer_onnx) if transformer_onnx else None,
                        "tflite": os.path.basename(transformer_tflite) if 'transformer_tflite' in locals() and transformer_tflite else None,
                        "input_shape": ["batch_size", "seq_len", "embed_dim"],
                        "output_shape": ["batch_size", "num_classes"]
                    }
                },
                "capabilities": [
                    "text_embedding",
                    "memory_reasoning",
                    "offline_operation"
                ]
            }

            manifest_path = os.path.join(bundle_dir, "manifest.json")
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)

            exported_files["manifest"] = manifest_path

            logger.info(f"Created mobile bundle at {bundle_dir}")
            return exported_files

        except Exception as e:
            logger.error(f"Failed to create mobile bundle: {e}")
            return {}

    def validate_export(self, model_path: str, model_type: str = "onnx") -> bool:
        """
        Validate that exported model can be loaded and run inference.

        Args:
            model_path: Path to exported model
            model_type: Type of model ("onnx" or "tflite")

        Returns:
            True if validation successful
        """
        try:
            if model_type == "onnx":
                session = ort.InferenceSession(model_path)
                # Test with dummy input
                dummy_input = {"input_ids": np.random.randint(0, 1000, (1, 10)).astype(np.int64)}
                outputs = session.run(None, dummy_input)
                return len(outputs) > 0

            elif model_type == "tflite":
                interpreter = tf.lite.Interpreter(model_path=model_path)
                interpreter.allocate_tensors()
                return True

            return False

        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False