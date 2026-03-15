import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoAdapterModel, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training


class TextEncoder(nn.Module):
    """Adapts a pre-trained SciBERT model for feature extraction."""

    def __init__(
        self, model_name: str, output_dim: int, peft_mode: str,
        lora_r: int, lora_alpha: int, lora_dropout: float,
        peft_target_modules: tuple[str, ...] | None,
        gradient_checkpointing: bool, freeze_backbone_until_layer: int,
        torch_dtype: torch.dtype | None = None, low_cpu_mem_usage: bool = True) -> None:
        super().__init__()
        peft_mode = peft_mode.lower()

        if model_name != "allenai/scibert":
            raise ValueError("This encoder is intended for SciBERT only.")

        # 4-bit Quantization (QLoRA)
        # NLP models are heavily memory-bound. QLoRA quantizes the frozen base model weights 
        # to a 4-bit NormalFloat (NF4) data type, slashing VRAM usage by ~75%, pair this 
        # with bfloat16 computation so the forward/backward passes and the tunable LoRA 
        # adapters maintain high precision and numerical stability.
        optype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        quantization_config = None
        
        if peft_mode == "qlora":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=optype)

        self.backbone_pretrained = AutoAdapterModel.from_pretrained(
            model_name, low_cpu_mem_usage=low_cpu_mem_usage,
            torch_dtype=torch_dtype, quantization_config=quantization_config)
        self.backbone = self.backbone_pretrained.load_adapter("allenai/scibert", source="hf", set_active=True)
        hidden_size = int(self.backbone.config.hidden_size)

        # Gradient Checkpointing
        # Standard backpropagation stores all intermediate activations in memory to calculate 
        # gradients later, which easily causes OOM errors on long sequences. 
        # Checkpointing drops these activations and recomputes them on-the-fly during the 
        # backward pass. It trades a ~20% increase in compute time for massive memory savings.
        if gradient_checkpointing and hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()
            if hasattr(self.backbone, "enable_input_require_grads"):
                self.backbone.enable_input_require_grads()

        if peft_mode == "qlora":
            self.backbone = prepare_model_for_kbit_training(self.backbone, use_gradient_checkpointing=gradient_checkpointing)

        if freeze_backbone_until_layer > 0:
            self.freeze_backbone(freeze_backbone_until_layer)

        # Low-Rank Adaptation (LoRA)
        # Instead of updating all 110M+ parameters of SciBERT, freeze the network and 
        # inject small, trainable low-rank matrices (A and B) into the Attention blocks. 
        # This reduces trainable parameters to less than 1%, preventing catastrophic 
        # forgetting of the base scientific vocabulary while drastically speeding up training.
        if peft_mode in {"lora", "qlora"}:
            target_modules = peft_target_modules or ("query", "value")
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                target_modules=list(target_modules), bias="none")
            self.backbone = get_peft_model(self.backbone, peft_config)

        self.projection = nn.Linear(hidden_size, output_dim) if hidden_size != output_dim else None

    def freeze_backbone(self, num_layers: int) -> None:
        """Freezes the embedding layer and the bottom N transformer blocks."""
        # Partial Backbone Freezing
        # In deep transformers, the lower layers learn universal, low-level syntactic 
        # features (grammar, word boundaries), while upper layers learn highly task-specific 
        # semantics. Freezing the bottom layers preserves the fundamental language model 
        # structure, prevents overfitting on small datasets, and accelerates training.
        base_model = getattr(self.backbone, getattr(self.backbone, "base_model_prefix", ""), self.backbone)
        embeddings = getattr(base_model, "embeddings", None)
        
        if embeddings is not None:
            for p in embeddings.parameters(): p.requires_grad = False
                
        encoder = getattr(base_model, "encoder", None)
        layers = getattr(encoder, "layer", None)
        
        if layers is not None:
            for layer in list(layers)[:max(0, num_layers)]:
                for p in layer.parameters():
                    p.requires_grad = False

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        hidden_states = self.backbone(input_ids, attention_mask).last_hidden_state
        # [CLS] Token Pooling
        # In BERT architectures, the special [CLS] token (at index 0) is designed to act 
        # as an aggregated sequence-level representation because it has full, unmasked 
        # bi-directional attention over the entire input sequence.
        cls_embedding = hidden_states[:, 0]
        if self.projection is not None:
            cls_embedding = self.projection(cls_embedding)
        return cls_embedding
