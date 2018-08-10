package com.algo.mlp;

import java.util.ArrayList;

public class Layer {
  private ArrayList<Neuron> neurons = new ArrayList<>();

  public Layer() {

  }

  public ArrayList<Neuron> getNeurons() {
    return neurons;
  }

  public void addNeuron(Neuron neuron){
    this.neurons.add(neuron);
  }
}