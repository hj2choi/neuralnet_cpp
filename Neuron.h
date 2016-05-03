
// link, that goes out from neuron.
// I had to realize that I don't really need this struct here.
struct Link {
	double weight;
	double prevDeltaWeight;
};


// neuron class
class Neuron {
public:
	double net; // net value on neuron
	double out; // output value of neuron ( out = activationFunc(net) )
	double deltaErr; // delta error value, for backward propagation
	std::vector<Link> links;

	// constructor for neuron: assign initial weight and create outgoing edges
	Neuron(unsigned numLinks) {
		for (int i=0; i<numLinks; ++i) {
			Link tempConnection;
			tempConnection.weight = (double)(std::rand()%100)/100;
			links.push_back(tempConnection);
			//std::cout << "created link for neuron" << std::endl;
		}
	}

};
