package com.algo.mlp;

import java.util.ArrayList;
import java.util.HashMap;

public class Neuron {

  private ArrayList<Connection> inputConnections = new ArrayList<>();
  private HashMap<Integer, Connection> inputConnectionsMap = new HashMap<>();
  private Connection biasConnection;
  private double output = 0d;
  private double bias = 1;
  private int id;

  public static int counter = 0;

  public Neuron() {
    counter++;
    id = counter;
  }

  public double activation(double x){
    return 1.0 / (1.0 +  (Math.exp(-x)));
  }

  public void calculateOutput(){
    output = 0;
    for(int i=0; i< inputConnections.size(); i++){
      output += inputConnections.get(i).getWeight() * inputConnections.get(i).getLeftNeuron().getOutput();
    }
    output += bias * biasConnection.getWeight();
    output = activation(output);
  }

  public void addInputConnection(ArrayList<Neuron> neurons){
    for(int i=0; i<neurons.size(); i++){
      Connection con = new Connection(neurons.get(i), this);
      inputConnections.add(con);
      inputConnectionsMap.put(neurons.get(i).getId(), con);
    }
  }

  public void addBiasConnection(Neuron neuron){
    Connection con = new Connection(neuron,this);
    biasConnection = con;
    inputConnections.add(con);
  }

  public Connection getConnection(int neuronIndex){
    return inputConnectionsMap.get(neuronIndex);
  }

  public ArrayList<Connection> getInputConnections() {
    return inputConnections;
  }

  public double getOutput() {
    return output;
  }

  public void setOutput(double output) {
    this.output = output;
  }

  public int getId() {
    return id;
  }
}
