r"""Example demo loading pre-trained language models.

Currently supports the following model types:
- BERT (bert-*) as a masked language model
- GPT-2 (gpt2* or distilgpt2) as a left-to-right language model

To run locally:
  python3 -m lit_nlp.examples.contrastive_gpt2_demo \
      --models=gpt2 --top_k 10 --port=5432

Then navigate to localhost:5432 to access the demo UI.
"""
import os
import sys
from typing import Optional, Sequence

from absl import app
from absl import flags
from absl import logging

from lit_nlp import dev_server
from lit_nlp import server_flags
from lit_nlp.api import model as lit_model
from lit_nlp.api import layout
from lit_nlp.api import types as lit_types
from lit_nlp.lib import utils
from lit_nlp.components import gradient_maps
from lit_nlp.examples.datasets import lm
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch


# NOTE: additional flags defined in server_flags.py

FLAGS = flags.FLAGS

FLAGS.set_default("development_demo", True)

_MODELS = flags.DEFINE_list(
    "models", ["gpt2"],
    "Models to load. Currently supports variants of GPT-2.")

_TOP_K = flags.DEFINE_integer(
    "top_k", 10, "Rank to which the output distribution is pruned.")

_MAX_EXAMPLES = flags.DEFINE_integer(
    "max_examples", 1000,
    "Maximum number of examples to load from each evaluation set. Set to None to load the full set."
)

_LOAD_BWB = flags.DEFINE_bool(
    "load_bwb", False,
    "If true, will load examples from the Billion Word Benchmark dataset. This may download a lot of data the first time you run it, so disable by default for the quick-start example."
)

# Custom frontend layout; see api/layout.py
modules = layout.LitModuleName
LM_LAYOUT = layout.LitCanonicalLayout(
    upper={
        "Main": [
            modules.EmbeddingsModule,
            modules.DataTableModule,
            modules.DatapointEditorModule,
            modules.SliceModule,
            modules.ColorModule,
        ]
    },
    lower={
        "Predictions": [
            modules.LanguageModelPredictionModule,
            modules.ConfusionMatrixModule,
        ],
        "Counterfactuals": [modules.GeneratorModule],
    },
    description="Custom layout for language models.",
)
CUSTOM_LAYOUTS = {"lm": LM_LAYOUT}

# You can also change this via URL param e.g. localhost:5432/?layout=default
FLAGS.set_default("default_layout", "lm")


def get_wsgi_app() -> Optional[dev_server.LitServerType]:
  FLAGS.set_default("server_type", "external")
  FLAGS.set_default("demo_mode", True)
  # Parse flags without calling app.run(main), to avoid conflict with
  # gunicorn command line flags.
  unused = flags.FLAGS(sys.argv, known_only=True)
  return main(unused)

class GPT2_LM(lit_model.Model):
  """GPT-2 LM model."""

  @property
  def num_layers(self):
    return self.model.config.n_layer


  def __init__(self, top_k=10):
    self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2", pad_token="<|endoftext|>")
    self.model = GPT2LMHeadModel.from_pretrained("gpt2")
    self.model.eval()
    self.top_k = top_k

  @staticmethod
  def clean_bpe_token(tok):
    if not tok.startswith("Ġ"):
      return "_" + tok
    else:
      return tok.replace("Ġ", "")

  def _detokenize(self, ids):
    tokens = self.tokenizer.convert_ids_to_tokens(ids)
    return [self.clean_bpe_token(t) for t in tokens]


  def _register_embedding_list_hook(self, model, embeddings_list): #TODO replace with self.model?
    def forward_hook(module, inputs, output):
        embeddings_list.append(output.squeeze(0).clone().cpu().detach().numpy())
    embedding_layer = model.transformer.wte
    handle = embedding_layer.register_forward_hook(forward_hook)
    return handle

  def _register_embedding_gradient_hooks(self, model, embeddings_gradients):
    def hook_layers(module, grad_in, grad_out):
        embeddings_gradients.append(grad_out[0].detach().cpu().numpy())
    embedding_layer = model.transformer.wte
    hook = embedding_layer.register_backward_hook(hook_layers)
    return hook

  def _pred(self, encoded_inputs):
    torch.enable_grad()
    embeddings_list = []
    handle = self._register_embedding_list_hook(self.model, embeddings_list)
    embeddings_gradients = []
    hook = self._register_embedding_gradient_hooks(self.model, embeddings_gradients)

    self.model.zero_grad()
    correct = encoded_inputs["input_ids"][-1]
    out = self.model(encoded_inputs["input_ids"], attention_mask=encoded_inputs["attention_mask"])
    model_probs = torch.nn.functional.softmax(out.logits, dim=-1)
    print(out.logits.shape)
    print(out.logits[0][-1][correct][-1])
    (out.logits[0][-1][correct][-1]).backward() # backward pass to obtain gradients

    handle.remove()
    hook.remove()

    top_k = model_probs.topk(self.top_k)
    batched_outputs = {
        "input_ids": encoded_inputs["input_ids"],
        "ntok": torch.sum(encoded_inputs["attention_mask"], dim=1),
        "top_k_indices": top_k.indices,
        "top_k_probs": top_k.values.detach(),
    }
    
    batched_outputs["input_embeddings"] = torch.Tensor(embeddings_list)
    batched_outputs["token_gradients"] = torch.Tensor(embeddings_gradients).squeeze(0)
    return batched_outputs

  def _postprocess(self, preds):
    """Post-process single-example preds. Operates on numpy arrays."""
    ntok = preds.pop("ntok")
    ids = preds.pop("input_ids")[:ntok]
    preds["tokens"] = self._detokenize(ids)

    # Decode predicted top-k tokens.
    # token_topk_preds will be a List[List[(word, prob)]]
    # Initialize prediction for 0th token as N/A.
    token_topk_preds = [[("N/A", 1.)]]
    pred_ids = preds.pop("top_k_indices")[:ntok]  # <int>[num_tokens, k]
    pred_probs = preds.pop("top_k_probs")[:ntok]  # <float32>[num_tokens, k]
    for token_pred_ids, token_pred_probs in zip(pred_ids, pred_probs):
      token_pred_words = self._detokenize(token_pred_ids)
      token_topk_preds.append(list(zip(token_pred_words, token_pred_probs)))
    preds["pred_tokens"] = token_topk_preds

    return preds

  ##
  # LIT API implementations
  def max_minibatch_size(self) -> int:
    # The lit.Model base class handles batching automatically in the
    # implementation of predict(), and uses this value as the batch size.
    return 6

  def predict_minibatch(self, inputs):
    """Predict on a single minibatch of examples."""
    # Preprocess inputs.
    texts = [ex["text"] for ex in inputs]
    encoded_inputs = self.tokenizer(texts, return_tensors="pt")

    # Get the predictions.
    batched_outputs = self._pred(encoded_inputs)
    # Convert to numpy for post-processing.
    for  k, v in batched_outputs.items():
        try:
            v.numpy()
        except:
            print(k,v)
    detached_outputs = {k: v.numpy() for k, v in batched_outputs.items()}
    # Split up batched outputs, then post-process each example.
    unbatched_outputs = utils.unbatch_preds(detached_outputs)
    return map(self._postprocess, unbatched_outputs)


  def input_spec(self):
    return {"text": lit_types.TextSegment(), "input_embeddings": lit_types.TokenEmbeddings()}


  def output_spec(self):
    spec = {
        # the "parent" keyword tells LIT which field in the input spec we should
        # compare this to when computing metrics.
        "pred_tokens": lit_types.TokenTopKPreds(align="tokens"),
        "tokens": lit_types.Tokens(parent="text"),  # all tokens
    }

    # Add input embeddings.
    spec["input_embeddings"] = lit_types.TokenEmbeddings()
    # Add model gradients.
    spec["token_gradients"] = lit_types.TokenGradients(align="tokens", grad_for="input_embeddings")
    return spec





def main(argv: Sequence[str]) -> Optional[dev_server.LitServerType]:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  ##
  # Load models, according to the --models flag.
  models = {}
  for model_name_or_path in _MODELS.value:
    # Ignore path prefix, if using /path/to/<model_name> to load from a
    # specific directory rather than the default shortcut.
    model_name = os.path.basename(model_name_or_path)
    if model_name.startswith("gpt2"):
      models[model_name] = GPT2_LM(
          model_name_or_path, top_k=_TOP_K.value)
    else:
      raise ValueError(
          f"Unsupported model name '{model_name}' from path '{model_name_or_path}'"
      )

  datasets = {
      # Empty dataset, if you just want to type sentences into the UI.
      "blank": lm.PlaintextSents(""),
  }

  for name in datasets:
    datasets[name] = datasets[name].slice[:_MAX_EXAMPLES.value]
    logging.info("Dataset: '%s' with %d examples", name, len(datasets[name]))

  generators = {"gradient_norm": gradient_maps.GradientNorm(), "gradient_dot_input": gradient_maps.GradientDotInput()}

  lit_demo = dev_server.Server(
      models,
      datasets,
      generators=generators,
      layouts=CUSTOM_LAYOUTS,
      **server_flags.get_flags())
  return lit_demo.serve()


if __name__ == "__main__":
  app.run(main)