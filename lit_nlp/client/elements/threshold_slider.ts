/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import {css, customElement, html, LitElement, property} from 'lit-element';
import {getMarginFromThreshold, getThresholdFromMargin} from '../lib/utils';
import {styles as sharedStyles} from '../modules/shared_styles.css';

/** A slider for setting classifcation thresholds/margins. */
@customElement('threshold-slider')
export class ThresholdSlider extends LitElement {
  @property({type: Number}) margin = 0;
  @property({type: String}) predKey = '';
  // Threshold sliders are between 0 and 1 for binary classification thresholds.
  // The other type of sliders are margin sliders between -5 and 5 for use with
  // mutliclass classifiers.
  @property({type: Boolean}) isThreshold = true;

  static get styles() {
    return [sharedStyles, css`
        .slider-row {
          display: flex;
          flex-wrap: wrap;
          margin: 8px 5px;
        }

        .slider-row div {
          padding-top: 9px;
        }

        .slider-label {
          width: 30px;
        }

        .hairline-button.reset-button {
          margin: 5px;
          padding: 5px 10px;
        }

    `];
  }

  renderThresholdSlider(margin: number, key: string) {
    // Convert between margin and classification threshold when displaying
    // margin as a threshold, as is done for binary classifiers.
    // Threshold is between 0 and 1 and represents the minimum score of the
    // positive (non-null) class before a datapoint is classified as positive.
    // A margin of 0 is the same as a threshold of .5 - meaning we take the
    // argmax class. A negative margin is a threshold below .5. Margin ranges
    // from -5 to 5, and can be converted the threshold through the equation
    // margin = ln(threshold / (1 - threshold)).
    const onChange = (e: Event) => {
      const newThresh = +(e.target as HTMLInputElement).value;
      const newMargin = getMarginFromThreshold(newThresh);
      const event = new CustomEvent('threshold-changed', {
        detail: {
          predKey: key,
          margin: newMargin
        }
      });
      this.dispatchEvent(event);
    };
    const marginToVal = (margin: number) => {
      const val = getThresholdFromMargin(+margin);
      return Math.round(100 * val) / 100;
    };
    return this.renderSlider(
        margin, key, 0, 1, 0.01, onChange, marginToVal, 'threshold');
  }

  renderMarginSlider(margin: number, key: string) {
    const onChange = (e: Event) => {
      const newMargin = (e.target as HTMLInputElement).value;
      const event = new CustomEvent('threshold-changed', {
        detail: {
          predKey: key,
          margin: newMargin
        }
      });
      this.dispatchEvent(event);
    };
    const marginToVal = (margin: number) => margin;
    return this.renderSlider(
        margin, key, -5, 5, 0.05, onChange, marginToVal, 'margin');
  }

  renderSlider(
      margin: number, key: string, min: number, max: number,
      step: number, onChange: (e: Event) => void,
      marginToVal: (margin: number) => number, title: string) {
    const val = marginToVal(margin);
    const isDefaultValue = margin === 0;
    const reset = (e: Event) => {
      const event = new CustomEvent('threshold-changed', {
        detail: {
          predKey: key,
          margin: 0
        }
      });
      this.dispatchEvent(event);
    };
    return html`
        <div class="slider-row">
          <div>${key} ${title}:</div>
          <input type="range" min="${min}" max="${max}" step="${step}"
                 .value="${val.toString()}" class="slider"
                 @change=${onChange}>
          <div class="slider-label">${val}</div>
          <button class='hairline-button reset-button' @click=${reset}
                  ?disabled="${isDefaultValue}">Reset</button>
        </div>`;
  }

  render() {
    return html`${this.isThreshold ?
        this.renderThresholdSlider(this.margin, this.predKey) :
        this.renderMarginSlider(this.margin, this.predKey)}`;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'threshold-slider': ThresholdSlider;
  }
}
