package com.algo.mlp;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

public class DataReader {

  private double[][] input;
  private double[][] expectedOutput;

  public DataReader(String filePath, int numberOfImages) throws IOException {

    input = new double[numberOfImages][784];
    expectedOutput = new double[numberOfImages][10];

    File file = new File(filePath);

    BufferedReader reader = new BufferedReader(new FileReader(file));
    String str;
    int count = 0;

    while((str=reader.readLine()) != null){
      String[] arr = str.split(",");

      for(int i=1; i<arr.length; i++){
        double num = Double.parseDouble(arr[i]);
        input[count][i-1] = num;
      }

      int out = Integer.parseInt(arr[0]);
      for(int i=0; i<10; i++)
        expectedOutput[count][i] = 0;
      expectedOutput[count][out] = 1;
      count++;
    }
  }

  public double[][] getInput() {
    return input;
  }

/*  public void setInput(double[][] input) {
    this.input = input;
  }*/

  public double[][] getExpectedOutput() {
    return expectedOutput;
  }

  /*public void setExpectedOutput(double[][] expectedOutput) {
    this.expectedOutput = expectedOutput;
  }*/

}