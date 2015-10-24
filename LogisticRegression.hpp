#ifndef LogisticRegression_hpp
#define LogisticRegression_hpp

#include "Dataset.hpp"

class LogisticRegression {
private:
    vector<double> weights;
    double bias;
    const DatasetMetadata* metadata;
    const vector<Instance*>& instances;
    double learningRate;
    int epochs;
    
    static double sigmoid(double x);
    double computeOutput(const Instance* instance) const;
    void updateWeights(const Instance* instance);
    void train();
    
public:
    LogisticRegression(const DatasetMetadata* metadata, const vector<Instance*>& instances, float learningRate, int epochs);
    
    const DatasetMetadata* getMetadata() const {
        return metadata;
    }
    
    string predict(const Instance* instance, double* confidence = 0) const;
    string toString() const;
};

#endif /* LogisticRegression_hpp */
