#include <cmath>
#include <sstream>

#include "LogisticRegression.hpp"

double LogisticRegression::sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double LogisticRegression::computeOutput(const Instance* instance) const {
    double net = bias;
    for (int i = 0; i < metadata->numOfFeatures; ++i) {
        net += weights[i] * instance->featureVector[i];
    }
    return sigmoid(net);
}

void LogisticRegression::updateWeights(const Instance* instance) {
    double output = computeOutput(instance);
    double delta = learningRate * (instance->classLabel - output) * output * (1.0 - output);
    bias += delta;
    for (int i = 0; i < metadata->numOfFeatures; ++i) {
        weights[i] += delta * instance->featureVector[i];
    }
}

void LogisticRegression::train() {
    weights.resize(metadata->numOfFeatures);
    for (int i = 0; i < weights.size(); ++i)
        weights[i] = 0.1;
    bias = 0.1;
    
    for (int i = 0; i < epochs; ++i) {
        for (int j = 0; j < instances.size(); ++j) {
            updateWeights(instances[j]);
        }
    }
}

LogisticRegression::LogisticRegression(const DatasetMetadata* metadata, const vector<Instance*>& instances, float learningRate, int epochs) : metadata(metadata), instances(instances), learningRate(learningRate), epochs(epochs) {
    train();
}

string LogisticRegression::predict(const Instance* instance, double* confidence) const {
    double output = computeOutput(instance);
    if (confidence)
        *confidence = output;
    
    output = output > 0.5 ? 1.0 : 0.0;
    return metadata->classVariable->convertInternalToValue(output);
}

string LogisticRegression::toString() const {
    stringstream ss;
    ss.setf(ios::fixed, ios::floatfield);
    ss.precision(6);
    
    for (int i = 0; i < weights.size(); ++i)
        ss << "weights[" << i << "] = " << weights[i] << endl;
    ss << "bias = " << bias << endl;
    
    return ss.str();
}