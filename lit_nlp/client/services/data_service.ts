/**
 * @license
 * Copyright 2022 Google LLC
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
import {action, computed, observable, reaction} from 'mobx';

import {ClassificationResult, IndexedInput, LitName, LitType, RegressionResult} from '../lib/types';
import {findSpecKeys, isLitSubtype} from '../lib/utils';

import {LitService} from './lit_service';
import {ApiService, AppState, ClassificationService, SettingsService} from './services';


/** Data source for a data column. */
export type Source = string;

/** Type for a data value. */
//tslint:disable-next-line:no-any
type ValueType = any;

/** Function type to set a column's data value for a new datapoint. **/
export type ValueFn = (input: IndexedInput) => ValueType;

/** Info about a data column. */
export interface DataColumnHeader {
  dataType: LitType;
  name: string;
  source: Source;
  getValueFn: ValueFn;
}

/** Types of columns auto calculated by data service. */
export enum CalculatedColumnType {
  PREDICTED_CLASS = "predicted class",
  CORRECT = "correct",
  ERROR = "error",
  SQUARED_ERROR = "squared error",
}

/** Map of datapoint ID to values for a column of data. */
export type ColumnData = Map<string, ValueType>;

/**
 * Data service singleton, responsible for maintaining columns of computed data
 * for datapoints in the current dataset.
 */
export class DataService extends LitService {
  @observable private readonly columnHeaders =
      new Map<string, DataColumnHeader>();
  @observable readonly columnData = new Map<string, ColumnData>();

  constructor(
      private readonly appState: AppState,
      private readonly classificationService: ClassificationService,
      private readonly apiService: ApiService,
      private readonly settingsService: SettingsService) {
    super();
    reaction(() => appState.currentDataset, () => {
      this.columnHeaders.clear();
      this.columnData.clear();
    });

    // Run classification interpreter when necessary.
    const getClassificationInputs = () =>
      [this.appState.currentInputData, this.appState.currentModels,
       this.classificationService.allMarginSettings];
    reaction(getClassificationInputs, () => {
      if (this.appState.currentInputData == null ||
          this.appState.currentInputData.length === 0 ||
              this.appState.currentModels.length === 0) {
        return;
      }
      if (!this.settingsService.isDatasetValidForModels(
              this.appState.currentDataset, this.appState.currentModels)) {
        return;
      }
      for (const model of this.appState.currentModels) {
        this.runClassification(model, this.appState.currentInputData);
      }
    }, {fireImmediately: true});

    // Run regression interpreter when necessary.
    const getRegressionInputs = () =>
      [this.appState.currentInputData, this.appState.currentModels];
    reaction(getRegressionInputs, () => {
      if (this.appState.currentInputData == null ||
          this.appState.currentInputData.length === 0 ||
              this.appState.currentModels.length === 0) {
        return;
      }
      if (!this.settingsService.isDatasetValidForModels(
              this.appState.currentDataset, this.appState.currentModels)) {
        return;
      }
      for (const model of this.appState.currentModels) {
        this.runRegression(model, this.appState.currentInputData);
      }
    }, {fireImmediately: true});

    this.appState.addNewDatapointsCallback(async (newDatapoints) =>
      this.setValuesForNewDatapoints(newDatapoints));
  }

  getColumnName(model: string, predKey: string, type?: CalculatedColumnType) {
    let columnName = `${model}:${predKey}`;
    if (type != null) {
      columnName += ` ${type}`;
    }
    return columnName;
  }

  /**
   * Run classification interpreter and store results in data service.
   */
  private async runClassification(model: string, data: IndexedInput[]) {
    const {output} = this.appState.currentModelSpecs[model].spec;
    if (findSpecKeys(output, 'MulticlassPreds').length === 0) {
      return;
    }

    const interpreterPromise = this.apiService.getInterpretations(
        data, model, this.appState.currentDataset, 'classification',
        this.classificationService.marginSettings[model],
        `Computing classification results`);
    const classificationResults = await interpreterPromise;

    // Add classification results as new columns to the data service.
    if (classificationResults != null && classificationResults.length) {
      const classificationKeys = Object.keys(classificationResults[0]);
      for (const key of classificationKeys) {
        // Parse results into new data columns and add the columns.
        const scoreFeatName = this.getColumnName(model, key);
        const predClassFeatName = this.getColumnName(
            model, key, CalculatedColumnType.PREDICTED_CLASS);
        const correctnessName = this.getColumnName(
            model, key, CalculatedColumnType.CORRECT);
        const scoreMap: ColumnData = new Map();
        const predClassMap: ColumnData = new Map();
        const correctnessMap: ColumnData = new Map();
        for (let i = 0; i < classificationResults.length; i++) {
          const input = this.appState.currentInputData[i];
          const result = classificationResults[i][key] as ClassificationResult;
          scoreMap.set(input.id, result.scores);
          predClassMap.set(input.id, result.predicted_class);
          correctnessMap.set(input.id, result.correct);
        }
        const source = `Classification:${model}`;
        this.addColumn(
            scoreMap, scoreFeatName, this.appState.createLitType(
                'MulticlassPreds', false), source);
        this.addColumn(
            predClassMap, predClassFeatName, this.appState.createLitType(
                'CategoryLabel', false), source);
        if (output[key].parent != null) {
          this.addColumn(
              correctnessMap, correctnessName, this.appState.createLitType(
                  'Boolean', false), source);
        }
      }
    }
  }

  /**
   * Run regression interpreter and store results in data service.
   */
  private async runRegression(model: string, data: IndexedInput[]) {
    const {output} = this.appState.currentModelSpecs[model].spec;
    if (findSpecKeys(output, 'RegressionScore').length === 0) {
      return;
    }

    const interpreterPromise = this.apiService.getInterpretations(
        data, model, this.appState.currentDataset, 'regression', undefined,
        `Computing regression results`);
    const regressionResults = await interpreterPromise;

    // Add regression results as new columns to the data service.
    if (regressionResults != null && regressionResults.length) {
      const regressionKeys = Object.keys(regressionResults[0]);
      for (const key of regressionKeys) {
        // Parse results into new data columns and add the columns.
        const scoreFeatName = this.getColumnName(model, key);
        const errorFeatName = this.getColumnName(
            model, key, CalculatedColumnType.ERROR);
        const sqErrorFeatName = this.getColumnName(
            model, key, CalculatedColumnType.SQUARED_ERROR);
        const scoreMap: ColumnData = new Map();
        const errorMap: ColumnData = new Map();
        const sqErrorMap: ColumnData = new Map();
        for (let i = 0; i < regressionResults.length; i++) {
          const input = this.appState.currentInputData[i];
          const result = regressionResults[i][key] as RegressionResult;
          scoreMap.set(input.id, result.score);
          errorMap.set(input.id, result.error);
          sqErrorMap.set(input.id, result.squared_error);
        }
        const dataType = this.appState.createLitType('Scalar', false);
        const source = `Regression:${model}`;
        this.addColumn(
            scoreMap, scoreFeatName, dataType, source);
        if (output[key].parent != null) {
          this.addColumn(
            errorMap, errorFeatName, dataType, source);
          this.addColumn(
            sqErrorMap, sqErrorFeatName, dataType, source);
        }
      }
    }
  }

  @action
  async setValuesForNewDatapoints(datapoints: IndexedInput[]) {
    // When new datapoints are created, set their data values for each
    // column stored in the data service.
    for (const input of datapoints) {
      for (const col of this.cols) {
        const key = col.name;
        const val = await this.columnHeaders.get(key)!.getValueFn(input);
        this.columnData.get(key)!.set(input.id, val);
      }
    }
  }

  @computed
  get cols(): DataColumnHeader[] {
    return Array.from(this.columnHeaders.values());
  }

  getColNamesOfType(typeName: LitName): string[] {
    return this.cols.filter(col => isLitSubtype(col.dataType, typeName)).map(
        col => col.name);
  }

  getColumnInfo(name: string): DataColumnHeader|undefined {
    return this.columnHeaders.get(name);
  }

  /** Flattened list of values in data columns for reacting to data changes. **/
  // TODO(b/156100081): Can we get observers to react to changes to columnData
  // without needing this computed list?
  @computed
  get dataVals() {
    const vals: ValueType[] = [];
    for (const colVals of this.columnData.values()) {
      vals.push(...colVals.values());
    }
    return vals;
  }

  /**
   * Add new column to data service, including values for existing datapoints.
   *
   * If column has been previously added, replaces the existing data with new
   * data, if they are different.
   */
  @action
  addColumn(
      columnVals: ColumnData, name: string, dataType: LitType, source: Source,
      getValueFn: ValueFn = () => null) {
    if (!this.columnHeaders.has(name)) {
      this.columnHeaders.set(name, {dataType, source, name, getValueFn});
    }
    if (!this.columnData.has(name) || (
            JSON.stringify(Array.from(columnVals.values())) !==
            JSON.stringify(Array.from(this.columnData.get(name)!.values())))) {
      this.columnData.set(name, columnVals);
    }
  }

  /** Get stored value for a datapoint ID for the provided column key. */
  getVal(id: string, key: string) {
    // If column not tracked by data service, get value from input data through
    // appState.
    if (!this.columnHeaders.has(key)) {
      return this.appState.getCurrentInputDataById(id)!.data[key];
    }
    // If no value yet stored for this datapoint for this column, return null.
    if (!this.columnData.get(key)!.has(id)) {
      return null;
    }
    return this.columnData.get(key)!.get(id);
  }

  /** Asyncronously get value for a datapoint ID for the provided column key.
   *
   *  This method is async as if the value has not yet been been retrieved
   *  for a new datapoint, it will return the promise fetching the value.
   */
  async getValAsync(id: string, key: string) {
    if (!this.columnHeaders.has(key) || this.columnData.get(key)!.has(id)) {
      return this.getVal(id, key);
    }

    const input = this.appState.getCurrentInputDataById(id)!;
    const val = await this.columnHeaders.get(key)!.getValueFn(input);
    this.columnData.get(key)!.set(input.id, val);
    return val;
  }

  /** Get list of column values from all datapoints. */
  getColumn(key: string): ValueType[] {
    // Map from the current input data, as opposed to getting from the data
    // service's columnData as the columnData might have some missing entries
    // for new datapoints where the value hasn't been asyncronously-returned.
    // This way, we ensure we get a list of values, one per datapoint, with
    // nulls for datapoints with no info for that column in the data service
    // yet.
    return this.appState.currentInputData.map(
        input => this.getVal(input.id, key));
  }
}
