package com.flonkings.ml.classify.logisticregression;
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
import org.apache.commons.math.FunctionEvaluationException;
import org.apache.commons.math.analysis.MultivariateRealFunction;
import org.apache.commons.math.linear.Array2DRowRealMatrix;
import org.apache.commons.math.linear.ArrayRealVector;
import org.apache.commons.math.linear.RealMatrix;

public final class SigmoidCost implements MultivariateRealFunction {

  private final Array2DRowRealMatrix X;
  private final double lambda;
  private final double[] yColumnVector;
  private final double m;
  private final double ratioPositive;
  private final double ratioNeg;

  public SigmoidCost(RealMatrix X, RealMatrix y, double lambda) {
    this.X = (Array2DRowRealMatrix) X;
    this.yColumnVector = y.getColumn(0);
    this.lambda = lambda;
    this.m = X.getRowDimension();
    int positive = 0;
    for (int i = 0; i < yColumnVector.length; i++) {
      if (yColumnVector[i] == 1) {
        positive++;
      }
    }
    ratioPositive = positive / (double)yColumnVector.length;
    ratioNeg = 1.0 - ratioPositive;
  }

  @Override
  public double value(double[] thetaArray) throws FunctionEvaluationException,
      IllegalArgumentException {
    /*
     * J = 1./m * (-y' * log(sigmoid(X*theta)) - (1 - y') * log(1-sigmoid(X*theta))) + lambda ./(2*m) * (theta(2:end)' * theta(2:end));
     */
    double sum = 0;
    final double[][] data = X.getDataRef();
    for (int i = 0; i < data.length; i++) {
      final double[] row = data[i];
      double current = 0.0d;
      double c = 0.0;
      for (int j = 0; j < row.length; j++) {
        double t = (row[j] * thetaArray[j]) - c;
        final double e = sum + t;
        c = (e - sum) - t;
        sum = e;
      }
      final double sig = (1.0d / (1.0d + Math.exp(-current)));
      sum += -yColumnVector[i] * Math.log1p(sig)
          - (1.0d - yColumnVector[i]) * Math.log1p(1.0d - sig);
    }
    double cost = (1.0d / m) * sum;
    double regSum = 0.0d;
    for (int i = 1; i < thetaArray.length; i++) {
      regSum += Math.pow(thetaArray[i], 2);
    }
    return cost + (regSum * (lambda / (2.0d * m)));
  }
  
 
  public static Predictor newPredictor(double[] thetas) {
    final ArrayRealVector vector = new ArrayRealVector(thetas, false);
    return new Predictor() {

      @Override
      public boolean predictDiscrete(double[] features) {
        return predict(features) >= 0.5d;
      }

      @Override
      public double predict(double[] features) {
        return 1.d / (1.d + Math.exp(-vector.dotProduct(features)));
      }

      @Override
      public boolean labelToDiscrete(double value) {
        return value >= 0.5d;
      }
    };
  }
}