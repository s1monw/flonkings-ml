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
import org.apache.commons.math.analysis.DifferentiableMultivariateRealFunction;
import org.apache.commons.math.analysis.MultivariateRealFunction;
import org.apache.commons.math.analysis.MultivariateVectorialFunction;
import org.apache.commons.math.linear.Array2DRowRealMatrix;
import org.apache.commons.math.optimization.GoalType;
import org.apache.commons.math.optimization.OptimizationException;
import org.apache.commons.math.optimization.RealConvergenceChecker;
import org.apache.commons.math.optimization.RealPointValuePair;
import org.apache.commons.math.optimization.general.AbstractScalarDifferentiableOptimizer;
import org.apache.commons.math.optimization.general.ConjugateGradientFormula;
import org.apache.commons.math.optimization.general.NonLinearConjugateGradientOptimizer;

import com.flonkings.ml.classify.CostFunction;
import com.flonkings.ml.classify.Model.DataSet;

public class LogisticRegression {
  
  private RealConvergenceChecker convergenceChecker;

  public LogisticRegression(RealConvergenceChecker convergenceChecker) {
    this.convergenceChecker = convergenceChecker;
  }
  
  public LogisticRegression() {
    this(new RealConvergenceChecker() {
      
      @Override
      public boolean converged(int iteration, RealPointValuePair previous,
          RealPointValuePair current) {
        if (Double.isNaN(current.getValue())) {
            throw new RuntimeException("NAN!");
        }
        return iteration >= 150 || current.getValue() == previous.getValue();
      }
    });
  }
  public CostFunction<MultivariateRealFunction,MultivariateVectorialFunction> createCostFunction(DataSet trainingSet, final double lambda) {
    final Array2DRowRealMatrix trainingMatrix = new Array2DRowRealMatrix(trainingSet.data);
    final Array2DRowRealMatrix lableMatrix = new Array2DRowRealMatrix(trainingSet.labels);
   
    return new CostFunction<MultivariateRealFunction, MultivariateVectorialFunction>() {

      @Override
      public MultivariateRealFunction newCostFunction() {
        return new SigmoidCost(trainingMatrix, lableMatrix, lambda);
      }

      @Override
      public MultivariateVectorialFunction newDerivative() {
        return new Gradient(trainingMatrix, lableMatrix, lambda);
      }
    };
  }
  
  
  public double[] train(CostFunction<MultivariateRealFunction,MultivariateVectorialFunction> function, double[] initialTheta) throws  FunctionEvaluationException, IllegalArgumentException, OptimizationException {
    final AbstractScalarDifferentiableOptimizer optimizer =  new NonLinearConjugateGradientOptimizer(ConjugateGradientFormula.FLETCHER_REEVES);
    optimizer.setMaxIterations(500);
    optimizer.setConvergenceChecker(this.convergenceChecker);
    final MultivariateRealFunction costFunction = function.newCostFunction();
    final MultivariateVectorialFunction derivativeFunction = function.newDerivative();
    
    final DifferentiableMultivariateRealFunction optimizeFunction = new DifferentiableMultivariateRealFunction() {
      @Override
      public double value(double[] point) throws FunctionEvaluationException,
          IllegalArgumentException {
        return costFunction.value(point);
      }

      @Override
      public MultivariateRealFunction partialDerivative(int k) {
        throw new UnsupportedOperationException("use #gradient() instead");
      }

      @Override
      public MultivariateVectorialFunction gradient() {
        return derivativeFunction;
      }
    };

    RealPointValuePair pair = optimizer.optimize(optimizeFunction, GoalType.MINIMIZE,
        initialTheta);
    return pair.getPoint();
  }
  
  public double predict(double[] features, double[] theta) {
    double sum = 0;
    for (int i = 0; i < theta.length; i++) {
      sum += features[i] * theta[i];
    }
    return  1.d / (1.d + Math.exp(-sum));

  }
  
  

  
}
