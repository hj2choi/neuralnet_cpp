#ifndef NET_H
#define NET_H

#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include "Neuron.h"

// activation function. it is currently implemented with sigmoid function
double activationFunc(double net);

class Net
{
public:

	/*
	    You should *not* change this part
	*/

	// constructor.
	// topology is a container representing net structure.
	//   e.g. {2, 4, 1} represents 2 neurons for the first layer, 4 for the second layer, 1 for last layer
	// if you want to hard-code the structure, just ignore the variable topology
	// eta: learning rate
	Net(const std::vector<unsigned> &topology, const double eta);

	// given an input sample inputVals, propagate input forward, compute the output of each neuron
	void feedForward(const std::vector<double> &inputVals);

	// given the vector targetVals (ground truth of output), propagate errors backward, and update each weights
	void backProp(const std::vector<double> &targetVals);

	// output the prediction for the current sample to the vector resultVals
	void getResults(std::vector<double> &resultVals) const;

	// return the error of the current sample
	double getError(void) const;


	/*
	    Add what you need in the below
	*/


	// ...

private:
	// ...
	std::vector<unsigned> topology;
	double eta;
	const static double alpha = 0.9; // momentum term, range: [0 1]

	std::vector< std::vector<Neuron> > *layers;

	const std::vector<double> *targetVals;
	//std::vector< std::vector<double> > *links;

	// activation function, (sigmoid function)


};



#endif//NET_H
