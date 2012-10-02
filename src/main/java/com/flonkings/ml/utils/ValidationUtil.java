package com.flonkings.ml.utils;

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
import com.flonkings.ml.classify.logisticregression.Predictor;

public class ValidationUtil {

  public static ValidationResult validate(DataSet set, Predictor predictor) {
    int numMissclassified = 0;
    final double[][] features = set.data;
    for (int i = 0; i < features.length; i++) {
      boolean prediction = predictor.predictDiscrete(features[i]);
      boolean expected = predictor.labelToDiscrete(set.labels[i]);
      if (prediction != expected) {
        numMissclassified++;
      }
    }
    double oneZeroError = (numMissclassified / (double) features.length);
    double accuracy = ((features.length - numMissclassified) / (double) features.length) * 100;
    return new ValidationResult(oneZeroError, accuracy);
  }

  public static class ValidationResult {
    public final double oneZeroError;
    public final double accuracy;

    ValidationResult(double oneZeroError, double accuracy) {
      this.oneZeroError = oneZeroError;
      this.accuracy = accuracy;
    }

    @Override
    public String toString() {
      return "ValidationResult [oneZeroError=" + oneZeroError + ", accuracy="
          + accuracy + "]";
    }

  }

}
