#include "Net.h"
using namespace std;

Net::Net(const std::vector<unsigned> &topology, const double eta, double (*atvfunc)(double), double (*atv_derv)(double)) {
  activation = atvfunc;
  atv_derivative = atv_derv;
  Net::topology=topology;
  Net::eta = eta;

  layers = new std::vector< std::vector<Neuron> >();
  //std::cout << "==============CREATE NEURONS WITH LAYER COUNT = " << topology.size() << std::endl;

  // for each layer
  for (int i=0; i<topology.size(); ++i) {
    std::vector<Neuron> temp;
    // for each unit in layer,
    for (int j=0; j<topology[i]; ++j) {
      std::cout << "neuron created, layer=" << i << ", unit="<< j << std::endl;

      int connectionCount;
      if (j<topology[i]-1) {
        connectionCount=topology[i+1];
      }
      Neuron unit(connectionCount);
      unit.net = j;

      temp.push_back(unit);

    }
    layers->push_back(temp);
    //std::cout << "information on layer just added: size=" << layers->back().size()<<endl;
  }


}

void Net::feedForward(const std::vector<double> &inputVals) {

  // feed values into input neurons
  for (int i=0; i<layers->at(0).size(); ++i) {
    layers->at(0).at(i).net = inputVals[i];
    layers->at(0).at(i).out = inputVals[i];
    //cout << "input at " << i << " =>" << layers->at(0).at(i).out << endl;
  }

  // feed forward, using weight values of preceding units.
  // update net and out values of each units
  for (int k=1; k<layers->size(); ++k) {
    for (int j=0;j<layers->at(k).size(); ++j) {
      double netValue=0;
      for (int i=0; i<layers->at(k-1).size(); ++i) {
        // update net value ====> net(j) += output(i) * weight(j)
        netValue += layers->at(k-1).at(i).out * layers->at(k-1).at(i).links.at(j).weight;
      }
      //std::cout << "feed forward neuron (" << k<< "," << j<< "): net ="<< netValue << ", out=" << activationFunc(netValue) << endl;
      layers->at(k).at(j).net = netValue;
      layers->at(k).at(j).out = activation(netValue);
    }
  }
}

void Net::backProp(const std::vector<double> &targetVals) {

  Net::targetVals = &targetVals;

  // for each output unit, compute deltaErr
  for (int i=0; i<layers->at(layers->size()-1).size(); ++i) {
    double deltaErr=0;
    double currentOut = layers->at(layers->size()-1).at(i).out;
    double currentNet = layers->at(layers->size()-1).at(i).net;
    deltaErr = atv_derivative(currentNet)*(targetVals[i]-currentOut);
    layers->at(layers->size()-1).at(i).deltaErr = deltaErr;
    //std::cout << "deltaErr = " << deltaErr << std::endl;
  }

  //cout << "===hidden delta Err=="<<endl;
  // for each layer, propogate backwards
  for (int k=layers->size()-2; k>=1; --k) {
    // for each hidden unit, compute deltaErr
    for (int i=0; i<layers->at(k).size(); ++i) {
      double deltaErr=0;
      double propagatedErr=0;
      double currentOut = layers->at(k).at(i).out;
      double currentNet = layers->at(k).at(i).net;

      // for each links towards hidden unit, calculate propogetedErr
      for (int j=0; j<layers->at(k+1).size(); ++j) {
        propagatedErr += (layers->at(k).at(i).links.at(j).weight) * (layers->at(k+1).at(j).deltaErr);
      }

      deltaErr = atv_derivative(currentNet)*propagatedErr;
      layers->at(k).at(i).deltaErr = deltaErr;
      //std::cout << "deltaErr = " << deltaErr << std::endl;
    }
  }


  // for each network links, update weight values
  for (int i=0; i<layers->size()-1; ++i) {
    for (int j=0; j<layers->at(i).size(); ++j) {
      // for each link
      for (int k=0; k<layers->at(i).at(j).links.size(); ++k) {
        double deltaWeight = eta * layers->at(i+1).at(k).deltaErr * layers->at(i).at(j).out;
        layers->at(i).at(j).links.at(k).weight += deltaWeight+ alpha*layers->at(i).at(j).links.at(k).prevDeltaWeight;
        layers->at(i).at(j).links.at(k).prevDeltaWeight = deltaWeight;
        cout << "new weight at (" << i<<","<< j<<") -> ("<< i+1<<","<<k <<") = "
          <<layers->at(i).at(j).links.at(k).weight <<
          "(+ " << deltaWeight << ")" << endl;
      }
    }
  }


}

void Net::getResults(std::vector<double> &resultVals) const {
  // copy values from last layer to resultVals
  resultVals.erase(resultVals.begin(), resultVals.end());
  for (int i=0; i<layers->at(layers->size()-1).size(); ++i) {
    resultVals.push_back(layers->at(layers->size()-1).at(i).out);
  }
}

double Net::getError(void) const {
  // use squared mean error
  double mean = 0;
  for (int i=0; i< layers->at(layers->size()-1).size(); ++i) {
    double currentOut = layers->at(layers->size()-1).at(i).out;
    mean += (targetVals->at(i)-currentOut)*(targetVals->at(i)-currentOut);
  }
  mean/=(layers->at(layers->size()-1).size());

  return mean;
}
