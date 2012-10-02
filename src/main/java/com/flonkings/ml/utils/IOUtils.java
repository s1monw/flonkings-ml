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
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.Charset;

import org.apache.lucene.util.ArrayUtil;

import com.flonkings.ml.classify.Model.DataSet;

public class IOUtils {

  
  public static DataSet loadCSVDataSet(InputStream stream) throws IOException {
    BufferedReader reader = new BufferedReader(new InputStreamReader(stream, Charset.forName("UTF-8")));
    String line;
    double[][] data = new double[1][];
    double[] labels = new double[1];
    int offset = 0;
    while((line = reader.readLine()) != null) {
      String[] split = line.split(",");
      if (labels.length == offset) {
        labels = ArrayUtil.grow(labels, offset+1);
        double[][] newData = new double[labels.length][];
        System.arraycopy(data, 0, newData, 0, data.length);
        data = newData;
      }
      data[offset] =  new double[split.length-1];
      for (int i = 0; i < split.length-1; i++) {
        data[offset][i] = Double.parseDouble(split[i]);
      }
      labels[offset++] = Double.parseDouble(split[split.length-1]);
    }
    return new DataSet(shrink(data, offset), shrink(labels, offset));
  }
  
  private static double[] shrink(double[] array, int targetSize) {
   
    if (targetSize != array.length) {
      double[] newArray = new double[targetSize];
      System.arraycopy(array, 0, newArray, 0, targetSize);
      return newArray;
    } else
      return array;
  }
  
  private static double[][] shrink(double[][] array, int targetSize) {
    if (targetSize != array.length) {
      double[][] newArray = new double[targetSize][];
      System.arraycopy(array, 0, newArray, 0, targetSize);
      return newArray;
    } else
      return array;
  }
}
