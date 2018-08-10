package com.algo.mlp;

public class Connection {

  private double weight = 0;
  private double prevDeltaWeight = 0;
  private Neuron leftNeuron, rightNeuron;

  public static int counter = 0;

  public Connection(Neuron leftNeuron, Neuron rightNeuron){
    this.leftNeuron = leftNeuron;
    this.rightNeuron = rightNeuron;
    counter++;
  }

  public double getWeight() {
    return weight;
  }

  public void setWeight(double weight) {
    this.weight = weight;
  }

  public double getPrevDeltaWeight() {
    return prevDeltaWeight;
  }

  public Neuron getLeftNeuron() {
    return leftNeuron;
  }

}