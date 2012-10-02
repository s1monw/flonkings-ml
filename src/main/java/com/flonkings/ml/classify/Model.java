package com.flonkings.ml.classify;

/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
import java.util.Random;

import com.flonkings.ml.utils.Utils;
import com.flonkings.ml.utils.Utils.NormalizationStats;

public class Model {

  private double[][] modelData;
  private double[] labels;
  private NormalizationStats normalizationStats;

  public Model(double[][] modelData, double[] labels, Random random) {
    this.modelData = modelData;
    shuffle(modelData, labels, random);
    this.labels = labels;
  }

  public void scaleFeatures(int pow) {
    for (int i = 0; i < modelData.length; i++) {
      modelData[i] = Utils.generatePolynomialFeatures(
          Utils.generateCombinatoricalFeatures(modelData[i]), pow);
    }
  }

  public void addOneFeature() {
    for (int i = 0; i < modelData.length; i++) {
      double[] newArray = new double[modelData[i].length + 1];
      newArray[0] = 1.0d;
      System.arraycopy(modelData[i], 0, newArray, 1, modelData[i].length);
      modelData[i] = newArray;
    }
  }

  public int featureSize() {
    return modelData[0].length;
  }

  public double[] getThetaVector() {
    return new double[featureSize()];
  }

  public NormalizationStats normalize() {
    this.normalizationStats = Utils.normalize(modelData, modelData.length,
        featureSize());
    Utils.applyNormalization(normalizationStats, modelData);
    return normalizationStats;
  }

  public NormalizationStats getNormalizationStats() {
    return normalizationStats;
  }

  public DataSet toDataSet() {
    return new DataSet(modelData, labels);
  }

  private static double[][] slice(double[][] source, int offset, int size) {
    double[][] data = new double[size][];
    System.arraycopy(source, offset, data, 0, size);
    return data;
  }

  private static double[] slice(double[] source, int offset, int size) {
    double[] data = new double[size];
    System.arraycopy(source, offset, data, 0, size);
    return data;
  }

  public static void shuffle(double[][] array, double[] labels, Random random) {
    int size = array.length;
    for (int i = size; i > 1; i--) {
      int randomElement = random.nextInt(i);
      double tmpLable = labels[randomElement];
      double[] tmp = array[randomElement];
      array[randomElement] = array[i - 1];
      labels[randomElement] = labels[i - 1];
      array[i - 1] = tmp;
      labels[i - 1] = tmpLable;
    }
  }

  public FoldIterator foldIterator(int folds) {
    return new FoldIterator(new DataSet(modelData, labels), folds);
  }

  public static class DataSet {
    public final double[][] data;
    public final double[] labels;

    public DataSet(double[][] model, double[] labels, int offset, int length) {
      this.data = slice(model, offset, length);
      this.labels = slice(labels, offset, length);
    }

    public DataSet(double[][] data, double[] labels) {
      this.data = data;
      this.labels = labels;
    }

  }
}
