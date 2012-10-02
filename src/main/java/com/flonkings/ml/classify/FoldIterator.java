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
import com.flonkings.ml.classify.Model.DataSet;

public class FoldIterator {
  private final double[][] data;
  private final double[] labels;
  private DataSet test;
  private DataSet training;
  private final int folds;
  private int foldId = 0;
  private final int length;

  public FoldIterator(DataSet set, int folds) {
    this.folds = folds;
    data = set.data;
    labels = set.labels;
    length = (int) (labels.length / (double) folds);
  }

  public boolean next() {
    if (foldId < folds) {
      int testOffset = length * foldId;
      int testLength = Math.min(data.length - testOffset, length);
      test = getTrainingSet(testOffset, testLength, 0, 0);
      int firstOffset = 0;
      int firstLength = testOffset - firstOffset;
      int secondOffset = (testOffset + testLength - 1);
      int secondLength = data.length - secondOffset;
      training = getTrainingSet(firstOffset, firstLength, secondOffset,
          secondLength);
      foldId++;
      return true;
    }
    return false;
  }

  private DataSet getTrainingSet(int offset, int length, int secondOffset,
      int secondLength) {
    double[][] dataArray = new double[length + secondLength][];
    double[] labelArray = new double[dataArray.length];
    if (length != 0) {
      System.arraycopy(data, offset, dataArray, 0, length);
      System.arraycopy(labels, offset, labelArray, 0, length);
    }
    if (secondLength != 0) {
      System.arraycopy(data, secondOffset, dataArray, length, secondLength);
      System.arraycopy(labels, secondOffset, labelArray, length, secondLength);
    }
    return new DataSet(dataArray, labelArray);

  }

  public DataSet getTestSet() {
    return this.test;
  }

  public DataSet getTrainingSet() {
    return this.training;
  }

}