#include <iostream>
#include <iomanip>

#include "LogisticRegression.hpp"

int main(int argc, const char * argv[]) {
    if (argc < 5) {
        cout << "usage: ./neuralnet dataset-file #folds learning-rate #epochs" << endl;
    } else {
        string datesetFile = argv[1];
        int folds = atoi(argv[2]);
        double learningRate = atof(argv[3]);
        int epochs = atoi(argv[4]);
        
        shared_ptr<Dataset> dataset(Dataset::loadDataset(datesetFile));
        const DatasetMetadata* metadata = dataset->getMetadata();
        
        const vector<Instance*> trainSet = dataset->getTrainSet();
        
        LogisticRegression lreg(metadata, trainSet, learningRate, epochs);
        cout << lreg.toString();
        
        const vector<Instance*>& testSet = trainSet;
        int correctCount = 0;
        cout << "<Predictions for the Test Set Instances>" << endl;
        for (int i = 0; i < testSet.size(); ++i) {
            Instance* inst = testSet[i];
            double confidence = 0.0;
            string predicted = lreg.predict(inst, &confidence);
            string actual = inst->toString(metadata, true);
            if (predicted == actual)
                correctCount++;
            cout << setfill(' ') << setw(3) << (i + 1) << ": ";
            cout << "Actual: " << actual << "  Predicted: " << predicted
                << "  Confidence: " << setprecision(6) << fixed << confidence << endl;
        }
        cout << "Number of correctly classified: " << correctCount << "  Total number of test instances: " << testSet.size() << endl;
    }
}