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
import java.util.ArrayList;
import java.util.List;

import com.flonkings.ml.classify.FoldIterator;
import com.flonkings.ml.classify.Model.DataSet;
import com.flonkings.ml.utils.Trainer.TrainingResult;

public abstract class Trainer<T extends TrainingResult> {

  public T train(FoldIterator iter) {
    List<T> results = new ArrayList<T>();
    while (iter.next()) {
      DataSet testSet = iter.getTestSet();
      DataSet trainingSet = iter.getTrainingSet();
      doTrain(trainingSet);
      T doValidate = doValidate(testSet);
      results.add(doValidate);
    }
    return accumulate(results);
  }

  protected abstract T accumulate(List<T> results);

  public abstract void doTrain(DataSet set);

  public abstract T doValidate(DataSet validationSet);

  public static class TrainingResult {
    public final double accuracy;

    public TrainingResult(double accuracy) {
      this.accuracy = accuracy;
    }

    public TrainingResult accumulate(TrainingResult... results) {
      double acc = 0.0d;
      for (TrainingResult trainingResult : results) {
        acc += trainingResult.accuracy;
      }
      return new TrainingResult(acc / (double) results.length);
    }

  }
}
