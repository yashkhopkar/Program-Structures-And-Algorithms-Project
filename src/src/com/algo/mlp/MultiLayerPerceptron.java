package com.algo.mlp;

import java.io.IOException;

public class MultiLayerPerceptron {


  public static String TRAINING_DATA_FILE_PATH = "com/algo/mlp/dataset/mnist_test.csv";
  public static String TESTING_DATA_FILE_PATH = "com/algo/mlp/dataset/mnist_train.csv";

  public static void main(String[] args) throws IOException {

    DataReader trainingDataReader = new DataReader(TRAINING_DATA_FILE_PATH, 60000);
    DataReader testingDataReader = new DataReader(TESTING_DATA_FILE_PATH, 10000);

    int[][] confusionMatrix = new int[10][10];

    int[] neurons = {
        784,  // 1st layer - input layer
        10,   // 2nd layer - hidden layer
        10    // 3rd layer - output layer
    };

    double[][] input = trainingDataReader.getInput();
    double[][] expectedOutput = trainingDataReader.getExpectedOutput();
    int numberOfTrainingImages = 60000;

    ArtificialNeuralNetwork artificialNeuralNetwork = new ArtificialNeuralNetwork(neurons, input, expectedOutput, 0.00000000001d);
    artificialNeuralNetwork.train(numberOfTrainingImages);

    input = testingDataReader.getInput();
    expectedOutput = testingDataReader.getExpectedOutput();

    artificialNeuralNetwork.setInput(input[0]);
    artificialNeuralNetwork.setExpectedOutput(expectedOutput);

    double[][] result = artificialNeuralNetwork.test(input);
    for (int i = 0; i < result.length; i++) {

      double maxValueForOutput = 0, indexOfOutput = 0;
      double maxValueForExpectedOutput = 0, indexOfExpectedOutput = 0;
      for (int j = 0; j < result[i].length; j++) {
        if (maxValueForOutput < result[i][j]) {
          maxValueForOutput = result[i][j];
          indexOfOutput = j;
        }
      }
      for (int j = 0; j < expectedOutput[i].length; j++) {

        if (maxValueForExpectedOutput < expectedOutput[i][j]) {
          maxValueForExpectedOutput = expectedOutput[i][j];
          indexOfExpectedOutput = j;
        }
      }

      confusionMatrix = updateConfusionMatrix(confusionMatrix, (int)indexOfOutput, (int)indexOfExpectedOutput);
    }

    printConfusionMatrix(confusionMatrix);

  }

  public static int[][] updateConfusionMatrix(int[][] matrix, int output, int target) {
    try {
      matrix[output][target]++;
    } catch (Exception e) {
    }
    return matrix;
  }


  public static void printConfusionMatrix(int[][] matrix) {
    System.out.println("==============Confusion Matrix - Start==============");
    for (int[] row : matrix) {
      for (int val : row) {
        System.out.print(val + "\t");
      }
      System.out.println();
    }
    System.out.println();
    System.out.println("==============Confusion Matrix - End==============");
  }


}
