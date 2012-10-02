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
import java.util.Arrays;

import org.apache.commons.math.FunctionEvaluationException;
import org.apache.commons.math.analysis.MultivariateVectorialFunction;
import org.apache.commons.math.linear.Array2DRowRealMatrix;
import org.apache.commons.math.linear.RealMatrix;

class Gradient implements MultivariateVectorialFunction {

  private final Array2DRowRealMatrix X;
  private final double[] y;
  private final Array2DRowRealMatrix X_transpose_normalized;
  private final double normalizedLambda;

  public Gradient(RealMatrix X, RealMatrix y, double lambda) {
    this.X = (Array2DRowRealMatrix) X;
    this.X_transpose_normalized = (Array2DRowRealMatrix)X.transpose().scalarMultiply(1.0d/X.getRowDimension());
    this.y = y.getColumn(0);
    this.normalizedLambda = lambda / X.getColumnDimension();
  }

  @Override
  public double[] value(double[] thetaArray)
      throws FunctionEvaluationException, IllegalArgumentException {
    /*
     * mask = ones(size(theta)); mask(1) = 0; grad = 1./m * X' *
     * (sigmoid(X*theta) - y) + (lambda ./ m) * (mask .* theta);
     */
    Array2DRowRealMatrix theta = new Array2DRowRealMatrix(thetaArray);
    Array2DRowRealMatrix multiply = ((Array2DRowRealMatrix)X).multiply(theta);
    double[][] column = multiply.getDataRef();
    for (int i = 0; i < column.length; i++) {
      double[] row = column[i];
      for (int j = 0; j < row.length; j++) {
        row[j] = (1.0d/(1.0d + Math.exp(-row[j]))) - y[i];
      }
    }
    Array2DRowRealMatrix g = X_transpose_normalized.multiply(multiply);
    double[] result = Arrays.copyOf(thetaArray, thetaArray.length);
    result[0] = 0;
    double[][] u = g.getDataRef();
    for (int i = 0; i < result.length; i++) {
      result[i] = u[i][0] + (result[i] * normalizedLambda);
    }
//    System.out.println(Arrays.toString(value1(thetaArray)));
//    System.out.println(Arrays.toString(result));

    return result;

  }
  
  public double[] value1(double[] thetaArray)
      throws FunctionEvaluationException, IllegalArgumentException {
    /*
     * mask = ones(size(theta)); mask(1) = 0; grad = 1./m * X' *
     * (sigmoid(X*theta) - y) + (lambda ./ m) * (mask .* theta);
     */
    double[] res = new double[X.getRowDimension()];
    double[][] column = X.getDataRef();
    for (int i = 0; i < column.length; i++) {
      double[] row = column[i];
      double current = 0.0d;
      for (int j = 0; j < row.length; j++) {
        current += row[j] * thetaArray[j];
      }
      res[i] = (1.0d/(1.0d + Math.exp(-current))) - y[i];
    }
    
    
    double[] result = Arrays.copyOf(thetaArray, thetaArray.length);
    double[][] dataRef = X_transpose_normalized.getDataRef();
    for (int i = 0; i < dataRef.length; i++) {
      double[] row = dataRef[i];
      double current = 0.0d;
      for (int j = 0; j < row.length; j++) {
        current += row[j] * res[j];
      }
      result[i] = current + (result[i] * normalizedLambda);
    }
    return result;

  }
  
  
  
}