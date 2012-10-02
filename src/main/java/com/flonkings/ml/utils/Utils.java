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
import java.util.Arrays;

import org.apache.commons.math.stat.StatUtils;
import org.apache.commons.math.util.FastMath;

public class Utils {

  public static double[] generatePolynomialFeatures(double[] init, int pow) {
    int newSize = init.length * (pow);
    double[] newArray = new double[newSize];
    for (int j = 0; j < pow; j++) {
      int offset = j * init.length;
      for (int i = 0; i < init.length; i++) {
        newArray[i + offset] = Math.pow(init[i], j + 1);
      }
    }

    return newArray;
  }

  public static double[][] generateCombinatoricalFeatures(double[][] data) {
    for (int i = 0; i < data.length; i++) {
      data[i] = generateCombinatoricalFeatures(data[i]);
    }
    return data;
  }

  public static double[] generateCombinatoricalFeatures(double[] data) {
    double[] newArray = new double[data.length * (data.length - 1)
        + data.length];
    int offset = data.length;
    System.arraycopy(data, 0, newArray, 0, data.length);
    for (int i = 0; i < data.length; i++) {
      for (int j = 0; j < data.length; j++) {
        if (i != j) {
          newArray[offset++] = data[i] * data[j];
        }
      }
    }
    return newArray;
  }

  public static NormalizationStats normalize(double[][] X) {
    return normalize(X, X.length, X[0].length);
  }

  public static NormalizationStats normalize(double[][] X, int m, int n) {
    double[] means = new double[n];
    double[] sigmas = new double[n];
    boolean[] isBinary = new boolean[n];
    Arrays.fill(isBinary, true);
    for (int j = 0; j < n; j++) {
      double[] col = new double[m];
      for (int i = 0; i < m; i++) {
        col[i] = X[i][j];
        if (X[i][j] != 0 && X[i][j] != 1) {
          isBinary[j] = false;
        }

      }
      means[j] = StatUtils.mean(col);
      sigmas[j] = FastMath.sqrt(StatUtils.variance(col, means[j]));
    }
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        if (isBinary[j]) {
          continue;
        }
        if (sigmas[j] == 0) {
          throw new IllegalArgumentException(
              "sigma must not be 0 for featrue: " + j);
        } else {
          X[i][j] = (X[i][j] - means[j]) / sigmas[j];
        }
      }
    }

    return new NormalizationStats(means, sigmas);
  }

  public static void applyNormalization(NormalizationStats stats, double[][] X) {
    for (int i = 0; i < X.length; i++) {
      applyNormalization(X[i], stats.getMeans(), stats.getVariance());
    }
  }

  public static double[] applyNormalization(double[] x, double[] mean,
      double[] variance) {
    for (int i = 0; i < x.length; i++) {
      x[i] = (x[i] - mean[i]) / variance[i];
    }
    return x;
  }

  public static class NormalizationStats {

    private double[] means;
    private double[] variance;

    private NormalizationStats(double[] means, double[] variance) {
      this.means = means;
      this.variance = variance;
    }

    public double[] getMeans() {
      return means;
    }

    public double[] getVariance() {
      return variance;
    }

    @Override
    public String toString() {
      return "NormalizationStats [means=" + Arrays.toString(means)
          + ", variance=" + Arrays.toString(variance) + "]";
    }
  }
}
