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
import java.io.IOException;
import java.io.InputStream;
import java.util.Random;

import junit.framework.Assert;

import org.apache.commons.math.FunctionEvaluationException;
import org.apache.commons.math.analysis.MultivariateRealFunction;
import org.apache.commons.math.analysis.MultivariateVectorialFunction;
import org.apache.commons.math.optimization.OptimizationException;
import org.junit.Test;

import com.flonkings.ml.classify.CostFunction;
import com.flonkings.ml.classify.Model;
import com.flonkings.ml.classify.Model.DataSet;
import com.flonkings.ml.utils.IOUtils;
import com.flonkings.ml.utils.ValidationUtil;

public class LogisticRegressionTest {

  
  @Test
  public void testLogReg() throws IOException, FunctionEvaluationException, IllegalArgumentException, OptimizationException {
    InputStream resourceAsStream = LogisticRegressionTest.class.getResourceAsStream("/classify_test_data.txt");
    DataSet dataSet = IOUtils.loadCSVDataSet(resourceAsStream);
    
    Model model = new Model(dataSet.data, dataSet.labels, new Random(1));
    model.scaleFeatures(2);
    model.normalize();
    model.addOneFeature();
    LogisticRegression logisticRegression = new LogisticRegression();
    DataSet trainingData = model.toDataSet();

    CostFunction<MultivariateRealFunction, MultivariateVectorialFunction> costFunc = logisticRegression
        .createCostFunction(trainingData, 0.3d);
    double[] solve = logisticRegression.train(costFunc, model.getThetaVector());
    Predictor predictor = SigmoidCost.newPredictor(solve);
    Assert.assertEquals(83.05084d, ValidationUtil.validate(trainingData, predictor).accuracy, 0.0001);
  }
}
