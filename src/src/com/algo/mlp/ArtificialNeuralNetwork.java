package com.algo.mlp;

import java.util.ArrayList;
import java.util.Random;

public class ArtificialNeuralNetwork {

  private ArrayList<Layer> layers = new ArrayList<>();
  private Random rand = new Random();
  private double epsilon;

  private Neuron bias = new Neuron();
  private double[][] input;
  private double[][] expectedOutput;

  public ArtificialNeuralNetwork(int[] neurons, double[][] input, double[][] expectedOutput, double epsilon) {
    this.input = input;
    this.expectedOutput = expectedOutput;
    this.epsilon = epsilon;

    for (int i = 0; i < neurons.length; i++) {
      Layer layer = new Layer();
      if (i == 0) {
        // input layer
        for (int j = 0; j < neurons[i]; j++) {
          Neuron neuron = new Neuron();
          layer.addNeuron(neuron);
        }
      } else if (i == neurons.length - 1) {
        // output layer
        for (int j = 0; j < neurons[i]; j++) {
          Neuron neuron = new Neuron();
          neuron.addBiasConnection(bias);
          neuron.addInputConnection(layers.get(i - 1).getNeurons());
          layer.addNeuron(neuron);
        }
      } else {
        // hidden layer
        for (int j = 0; j < neurons[i]; j++) {
          Neuron neuron = new Neuron();
          neuron.addBiasConnection(bias);
          neuron.addInputConnection(layers.get(i - 1).getNeurons());
          layer.addNeuron(neuron);
        }
      }
      layers.add(layer);
    }

    // initialise random weight
    for (int i = 1; i < layers.size(); i++) {
      Layer layer = layers.get(i);
      for (int j = 0; j < layer.getNeurons().size(); j++) {
        Neuron neuron = layer.getNeurons().get(j);
        for (int k = 0; k < neuron.getInputConnections().size(); k++) {
          Connection connection = neuron.getInputConnections().get(k);
          connection.setWeight(-0.5 + (0.5 - (-0.5)) * rand.nextDouble());
        }
      }
    }

    Neuron.counter = 0;
    Connection.counter = 0;
  }

  public double[][] test(double[][] input) {
    double[][] result = new double[input.length][];

    System.out.println("Prediction on test data started");
    for (int i = 0; i < input.length; i++) {
      setInput(input[i]);

      calculateOutputForEachLayer();
      double[] output = getOutput();
      result[i] = output;

    }
    System.out.println("Prediction on test data ended");

    return result;
  }

  public void train(int numberOfImages) {
    System.out.println("Training neural network started");
    for (int i = 0; i < numberOfImages; i++) {

      setInput(input[i]);

      calculateOutputForEachLayer();

      backPropagation(expectedOutput[i]);

    }
    System.out.println("Training neural network ended");
  }

  public double[] getOutput() {
    // getting output from the output layer
    double[] outputs = new double[layers.get(layers.size() - 1).getNeurons().size()];
    for (int i = 0; i < layers.get(layers.size() - 1).getNeurons().size(); i++)
      outputs[i] = layers.get(layers.size() - 1).getNeurons().get(i).getOutput();
    return outputs;
  }

  public void setInput(double input[]) {
    // set input
    for (int i = 0; i < layers.get(0).getNeurons().size(); i++) {
      Neuron neuron = layers.get(0).getNeurons().get(i);
      neuron.setOutput(input[i]);
    }
  }

  public void setExpectedOutput(double[][] expectedOutput) {
    this.expectedOutput = expectedOutput;
  }

  public double normalize(double x) {
    if (x < 0) {
      return epsilon;
    } else if (x > 1)
      return 1 - epsilon;
    else
      return x;
  }

  public void calculateOutputForEachLayer() {
    for (int i = 1; i < layers.size(); i++) {
      for (int j = 0; j < layers.get(i).getNeurons().size(); j++) {
        layers.get(i).getNeurons().get(j).calculateOutput();
      }
    }
  }

  public void backPropagation(double[] expectedOutput) {
    for (int i = 0; i < expectedOutput.length; i++) {
      expectedOutput[i] = normalize(expectedOutput[i]);
    }

    Layer outputLayer = layers.get(layers.size() - 1);
    for (int i = 0; i < outputLayer.getNeurons().size(); i++) {
      Neuron neuron = outputLayer.getNeurons().get(i);
      ArrayList<Connection> inputConnections = neuron.getInputConnections();
      for (int j = 0; j < inputConnections.size(); j++) {
        Connection inputConnection = inputConnections.get(j);

        double neuronOutput = neuron.getOutput();
        double leftNeuronOutput = inputConnections.get(j).getLeftNeuron().getOutput();
        double desiredOutput = expectedOutput[i];

        double weightMultiplier = -neuronOutput * (1 - neuronOutput) * leftNeuronOutput * (desiredOutput - neuronOutput);
        double newWeight = inputConnection.getWeight() + weightMultiplier;
        inputConnection.setWeight(newWeight + inputConnection.getPrevDeltaWeight());
      }
    }

    // updating weight in hidden layers
    for (int i = 1; i < layers.size() - 1; i++) {
      Layer layer = layers.get(i);
      for (int j = 0; j < layer.getNeurons().size(); j++) {
        Neuron neuron = layer.getNeurons().get(j);
        ArrayList<Connection> inputConnections = neuron.getInputConnections();
        for (int k = 0; k < inputConnections.size(); k++) {
          Connection inputConnection = inputConnections.get(k);
          double neuronOutput = neuron.getOutput();
          double leftNeuronOutput = inputConnection.getLeftNeuron().getOutput();
          double sumOfAllOutputLayerNeuronOutputs = 0;

          for (int l = 0; l < outputLayer.getNeurons().size(); l++) {
            Neuron outputNeuron = outputLayer.getNeurons().get(l);
            double outputNeuronWeight = outputNeuron.getConnection(neuron.getId()).getWeight();
            double desiredOutput = expectedOutput[l];
            double outputNeuronOutput = outputNeuron.getOutput();
            sumOfAllOutputLayerNeuronOutputs += -(desiredOutput - outputNeuronOutput) * outputNeuronOutput * (1 - outputNeuronOutput) * outputNeuronWeight;
          }
          double weightMultiplier = neuronOutput * (1 - neuronOutput) * leftNeuronOutput * sumOfAllOutputLayerNeuronOutputs;
          double newWeight = inputConnection.getWeight() + weightMultiplier;
          inputConnection.setWeight(newWeight + inputConnection.getPrevDeltaWeight());
        }
      }
    }
  }
}