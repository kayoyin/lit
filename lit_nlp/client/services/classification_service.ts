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

// tslint:disable:no-new-decorators
import {action, computed, observable} from 'mobx';
import {FacetedData, GroupedExamples} from '../lib/types';
import {LitService} from './lit_service';

/**
 * A margin setting is the margin value and the facet information for which
 * datapoints from a dataset that margin value applies to.
 */
interface MarginSetting {
  facetData?: FacetedData;
  margin: number;
}

/**
 * Any facet of a dataset can have its own margin value. Key string represents
 * the facet key/value pairs.
 */
interface MarginsPerFacet {
  [facetString: string]: MarginSetting;
}

/** Each output field has its own margin settings. */
export interface MarginsPerField {
  [fieldName: string]: MarginsPerFacet;
}

/**
 * Classification margin settings across all models and prediction heads.
 *
 * Margins are a generalized way to define classification thresholds beyond
 * binary classification score threshoods.
 */
export interface MarginSettings {
  [model: string]: MarginsPerField;
}

/**
 * A singleton class that stores margin settings for classification tasks,
 * including margins for dataset facets.
 */
export class ClassificationService extends LitService {
  @observable marginSettings: MarginSettings = {};

  constructor() {
    super();
  }

  // Returns all margin settings for use as a reaction input function when
  // setting up observers.
  // TODO(lit-team): Remove need for this intermediate object (b/156100081)
  @computed
  get allMarginSettings(): number[] {
    const res: number[] = [];
    for (const settingsPerModel of Object.values(this.marginSettings)) {
      for (const settingsPerPredKey of Object.values(settingsPerModel)) {
         for (const settingsPerFacet of Object.values(settingsPerPredKey)) {
           res.push(settingsPerFacet.margin);
         }
      }
    }
    return res;
  }

  /**
   * Reset the facet groups that store margins for a field based on the
   * facets from the groupedExamples.
   */
  @action
  setMarginGroups(model: string, fieldName: string,
                  groupedExamples: GroupedExamples) {
    if (this.marginSettings[model] == null) {
      this.marginSettings[model] = {};
    }
    this.marginSettings[model][fieldName] = {};
    for (const group of Object.values(groupedExamples)) {
      this.marginSettings[model][fieldName][group.displayName!] =
          {facetData: group, margin: 0};
    }
  }

  @action
  setMargin(model: string, fieldName: string, value: number,
            facet?: FacetedData) {
    if (this.marginSettings[model] == null) {
      this.marginSettings[model] = {};
    }
    if (this.marginSettings[model][fieldName] == null) {
      this.marginSettings[model][fieldName] = {};
    }
    if (facet == null) {
      // If no facet provided, then update the facet for the entire dataset
      // if one exists, otherwise update all facets with the provided margin.
      if ("" in this.marginSettings[model][fieldName]) {
        this.marginSettings[model][fieldName][""] =
            {facetData: facet, margin: value};
      } else {
        for (const key of Object.keys(this.marginSettings[model][fieldName])) {
          this.marginSettings[model][fieldName][key].margin = value;
        }
      }
    } else {
      this.marginSettings[model][fieldName][facet.displayName!] =
          {facetData: facet, margin: value};
    }
  }

  getMargin(model: string, fieldName: string, facet?: FacetedData) {
    if (this.marginSettings[model] == null ||
        this.marginSettings[model][fieldName] == null) {
      return 0;
    }
    if (facet == null) {
      if (this.marginSettings[model][fieldName][""] == null) {
        return 0;
      }
      return this.marginSettings[model][fieldName][""].margin;
    } else {
      if (this.marginSettings[model][fieldName][facet.displayName!] == null) {
        return 0;
      }
      return this.marginSettings[model][fieldName][facet.displayName!].margin;
    }
  }
}
