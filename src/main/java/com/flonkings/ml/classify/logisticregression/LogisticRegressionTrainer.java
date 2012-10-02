package com.flonkings.ml.classify.logisticregression;

import java.util.List;

import org.apache.commons.math.analysis.MultivariateRealFunction;
import org.apache.commons.math.analysis.MultivariateVectorialFunction;

import com.flonkings.ml.classify.CostFunction;
import com.flonkings.ml.classify.FoldIterator;
import com.flonkings.ml.classify.Model;
import com.flonkings.ml.classify.Model.DataSet;
import com.flonkings.ml.utils.Trainer;
import com.flonkings.ml.utils.ValidationUtil;
import com.flonkings.ml.utils.ValidationUtil.ValidationResult;

public class LogisticRegressionTrainer extends
    Trainer<LogisticRegressionTraingingResult> {

  private LogisticRegression logReg;
  private Model model;
  private double lambda;
  private double[] trainedThetas;
  private final boolean adjustLambda;

  public LogisticRegressionTrainer(LogisticRegression logReg, Model model) {
    this(logReg, model, true, 0.1);
  }

  private LogisticRegressionTrainer(LogisticRegression logReg, Model model,
      boolean adjustLambda, double lambda) {
    this.logReg = logReg;
    this.adjustLambda = adjustLambda;
    this.model = model;
    this.lambda = lambda;
  }

  @Override
  public void doTrain(DataSet set) {
    if (adjustLambda) {
      double accur = 0.0d;
      for (int i = -6; i < 6; i++) {
        double lambda = Math.pow(2, i);
        System.out.println("train with lambda: " + lambda);

        LogisticRegressionTrainer logisticRegressionTrainer = new LogisticRegressionTrainer(
            this.logReg, model, false, lambda);
        LogisticRegressionTraingingResult res = logisticRegressionTrainer
            .train(new FoldIterator(set, 5));
        System.out.println(res.accuracy);
        if (res.accuracy > accur) {
          accur = res.accuracy;
          this.lambda = lambda;
        }

      }
      System.out.println(this.lambda);

    }
    try {
      CostFunction<MultivariateRealFunction, MultivariateVectorialFunction> costFunc = logReg
          .createCostFunction(set, lambda);
      trainedThetas = logReg.train(costFunc, model.getThetaVector());
    } catch (Exception ex) {
      throw new RuntimeException(ex);
    }

  }

  @Override
  public LogisticRegressionTraingingResult doValidate(DataSet validationSet) {
    ValidationResult validate = ValidationUtil.validate(validationSet,
        SigmoidCost.newPredictor(trainedThetas));
    return new LogisticRegressionTraingingResult(validate.accuracy,
        validate.oneZeroError, this.lambda);
  }

  @Override
  protected LogisticRegressionTraingingResult accumulate(
      List<LogisticRegressionTraingingResult> results) {
    if (results.size() == 1) {
      return results.get(0);
    }
    double lambda = 0.0;
    double oneZeroError = Double.MAX_VALUE;
    double accuracy = 0.0d;
    for (LogisticRegressionTraingingResult logisticRegressionTraingingResult : results) {
      lambda += logisticRegressionTraingingResult.lambda;
      oneZeroError += logisticRegressionTraingingResult.oneZeroError;
      accuracy += logisticRegressionTraingingResult.accuracy;
    }
    return new LogisticRegressionTraingingResult(accuracy
        / (double) results.size(), oneZeroError / (double) results.size(),
        lambda / (double) results.size());
  }

  public double[] train(LogisticRegressionTraingingResult result, DataSet set) {
    try {
      CostFunction<MultivariateRealFunction, MultivariateVectorialFunction> costFunc = logReg
          .createCostFunction(set, result.lambda);
      return logReg.train(costFunc, model.getThetaVector());
    } catch (Exception ex) {
      throw new RuntimeException(ex);
    }
  }

}
