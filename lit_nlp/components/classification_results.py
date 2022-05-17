# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""An interpreter for analyzing classification results."""

import numbers
from typing import cast, Dict, List, Optional, Sequence, Text

from lit_nlp.api import components as lit_components
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import dtypes
from lit_nlp.api import model as lit_model
from lit_nlp.api import types
from lit_nlp.lib import utils as lit_utils
import numpy as np

JsonDict = types.JsonDict
IndexedInput = types.IndexedInput
Spec = types.Spec


def get_margin_for_input(margin_config: Optional[JsonDict] = None,
                         inp: Optional[JsonDict] = None):
  """Get margin given a margin config and input example."""
  if not margin_config:
    return 0

  for margin_entry in margin_config.values():
    facet_info = (margin_entry['facetData']['facets']
                  if 'facetData' in margin_entry else {})
    match = True
    if inp:
      for feat, facet_info in facet_info.items():
        value = facet_info['val']
        if (isinstance(inp[feat], numbers.Number) and
            not isinstance(inp[feat], bool)):
          # If the facet is a numeric range string, extract the min and max
          # and check the value against that range.
          min_val = value[0]
          max_val = value[1]
          if not (inp[feat] >= min_val and inp[feat] < max_val):
            match = False
        # If the facet is a standard value, check the feature value for
        # equality to it.
        elif inp[feat] != value:
          match = False
    if match:
      return margin_entry['margin']
  return 0


def get_classifications(
    preds: Sequence[np.ndarray], pred_spec: types.MulticlassPreds,
    margin_config: Optional[Sequence[float]] = None) -> Sequence[str]:
  """Get classified indices given prediction scores and configs."""
  # If there is a margin set for the prediction, take the log of the prediction
  # scores and add the margin to the null indexes value before taking argmax
  # to find the predicted class.
  if margin_config is not None:
    multiclass_pred_spec = cast(types.MulticlassPreds, pred_spec)
    null_idx = multiclass_pred_spec.null_idx
    pred_idxs = []
    for p, margin in zip(preds, margin_config):
      logit_mask = margin * np.eye(len(multiclass_pred_spec.vocab))[null_idx]
      pred_idx = np.argmax(np.log(p) + logit_mask)
      pred_idxs.append(pred_idx)
  else:
    pred_idxs = [np.argmax(p) for p in preds]
  return [pred_spec.vocab[idx] for idx in pred_idxs]


class ClassificationInterpreter(lit_components.Interpreter):
  """Calculates and returns classification results, using thresholds."""

  def run(self,
          inputs: List[JsonDict],
          model: lit_model.Model,
          dataset: lit_dataset.IndexedDataset,
          model_outputs: Optional[List[JsonDict]] = None,
          config: Optional[JsonDict] = None):

    # Find the prediction field key in the model output to use for calculations.
    output_spec = model.output_spec()
    supported_keys = self._find_supported_pred_keys(output_spec)

    results: List[Dict[Text, dtypes.ClassificationResult]] = []

    # Run prediction if needed:
    if model_outputs is None:
      model_outputs = list(model.predict(inputs))

    for i, inp in enumerate(inputs):
      input_result: Dict[Text, dtypes.ClassificationResult] = {}
      for key in supported_keys:
        if isinstance(model_outputs[i][key], dtypes.ClassificationResult):
          continue

        margin = get_margin_for_input(
            config[key] if (config and key in config) else None, inp)
        scores = model_outputs[i][key]
        pred_class = get_classifications(
            [scores], output_spec[key], [margin])[0]
        correct = None
        # If there is ground truth information, calculate error and squared
        # error.
        if (output_spec[key].parent and
            output_spec[key].parent in inp):
          correct = pred_class == inp[output_spec[key].parent]

        result = dtypes.ClassificationResult(scores, pred_class, correct)
        input_result[key] = result
      results.append(input_result)
    return results

  def is_compatible(self, model: lit_model.Model) -> bool:
    output_spec = model.output_spec()
    return True if self._find_supported_pred_keys(output_spec) else False

  def _find_supported_pred_keys(self, output_spec: types.Spec) -> List[Text]:
    return lit_utils.find_spec_keys(output_spec, types.MulticlassPreds)
