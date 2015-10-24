#include <iostream>
#include <iomanip>
#include <memory>
#include <chrono>
#include <random>
#include <algorithm>

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
        
        const vector<Instance*> data = dataset->getTrainSet();
        
        if (folds <= 1) {
            LogisticRegression lreg(metadata, data, learningRate, epochs);
            
            int correctCount = 0;
            int count = (int)data.size();
            for (int i = 0; i < count; ++i) {
                Instance* inst = data[i];
                double confidence = 0.0;
                string predicted = lreg.predict(inst, &confidence);
                string actual = inst->toString(metadata, true);
                if (predicted == actual)
                    correctCount++;
                
                cout << setfill(' ') << setw(3) << (i + 1) << ": ";
                cout << "Actual: " << actual << "  Predicted: " << predicted;
                cout << "  Confidence: " << setprecision(6) << fixed << confidence << endl;
            }
            
            cout << "Training-Set Accuracy: " << setfill(' ') << setw(3) << correctCount << " / "
                << setfill(' ') << setw(3) << count
                << " ( " << setprecision(4) << fixed << (100.0 * correctCount / count) << "% )" << endl;
        } else {
            vector<int> dataIdxPos;
            vector<int> dataIdxNeg;
            for (int i = 0; i < data.size(); ++i) {
                if (data[i]->classLabel > 0.5)
                    dataIdxPos.push_back(i);
                else
                    dataIdxNeg.push_back(i);
            }
            unsigned int seed = (unsigned int)chrono::system_clock::now().time_since_epoch().count();
            shuffle (dataIdxPos.begin(), dataIdxPos.end(), default_random_engine(seed));
            shuffle (dataIdxNeg.begin(), dataIdxNeg.end(), default_random_engine(seed));
            
            vector<int> foldIdx(data.size());
            vector<string> actualLabel(data.size());
            vector<string> predictedLabel(data.size());
            vector<double> predictedConfidence(data.size());
            
            int correctCountTrainSum = 0;
            int correctCountTestSum = 0;
            int countTrainSum = 0;
            int countTestSum = 0;
            
            for (int i = 0; i < folds; ++i) {
                // stratified sampling
                int beginPos = (int)dataIdxPos.size() * i / folds;
                int endPos = (int)dataIdxPos.size() * (i + 1) / folds - 1;
                int beginNeg = (int)dataIdxNeg.size() * i / folds;
                int endNeg = (int)dataIdxNeg.size() * (i + 1) / folds - 1;
                
                vector<int> trainSetIdx;
                vector<int> testSetIdx;
                for (int j = 0; j < dataIdxPos.size(); ++j) {
                    if (j >= beginPos && j <= endPos)
                        testSetIdx.push_back(dataIdxPos[j]);
                    else
                        trainSetIdx.push_back(dataIdxPos[j]);
                }
                for (int j = 0; j < dataIdxNeg.size(); ++j) {
                    if (j >= beginNeg && j <= endNeg)
                        testSetIdx.push_back(dataIdxNeg[j]);
                    else
                        trainSetIdx.push_back(dataIdxNeg[j]);
                }
                shuffle (trainSetIdx.begin(), trainSetIdx.end(), default_random_engine(seed));
                shuffle (testSetIdx.begin(), testSetIdx.end(), default_random_engine(seed));
                
                // training
                vector<Instance*> trainSet(trainSetIdx.size());
                for (int j = 0; j < trainSetIdx.size(); ++j)
                    trainSet[j] = data[trainSetIdx[j]];
                
                LogisticRegression lreg(metadata, trainSet, learningRate, epochs);
                
                // predicting
                int correctCountTrain = 0;
                int countTrain = (int)trainSetIdx.size();
                for (int j = 0; j < countTrain; ++j) {
                    Instance* inst = data[trainSetIdx[j]];
                    string predicted = lreg.predict(inst);
                    string actual = inst->toString(metadata, true);
                    if (predicted == actual)
                        correctCountTrain++;
                }
                int correctCountTest = 0;
                int countTest = (int)testSetIdx.size();
                for (int j = 0; j < countTest; ++j) {
                    Instance* inst = data[testSetIdx[j]];
                    double confidence = 0.0;
                    string predicted = lreg.predict(inst, &confidence);
                    string actual = inst->toString(metadata, true);
                    if (predicted == actual)
                        correctCountTest++;
                    
                    foldIdx[testSetIdx[j]] = i + 1;
                    actualLabel[testSetIdx[j]] = actual;
                    predictedLabel[testSetIdx[j]] = predicted;
                    predictedConfidence[testSetIdx[j]] = confidence;
                }
                
                correctCountTrainSum += correctCountTrain;
                correctCountTestSum += correctCountTest;
                countTrainSum += countTrain;
                countTestSum += countTest;
                
                cout << "Fold " << setfill(' ') << setw(2) << (i + 1) << ": ";
                cout << "Training-Set Accuracy: " << setfill(' ') << setw(3) << correctCountTrain << " / "
                    << setfill(' ') << setw(3) << countTrain << "  ";
                cout << "Test-Set Accuracy: " << setfill(' ') << setw(2) << correctCountTest << " / "
                    << setfill(' ') << setw(2) << countTest << endl;
            }
            
            cout << "Average: ";
            cout << "Training-Set Accuracy: " << setprecision(4) << fixed
                << (100.0 * correctCountTrainSum / countTrainSum) << "%   ";
            cout << "Test-Set Accuracy: " << setprecision(4) << fixed
                << (100.0 * correctCountTestSum / countTestSum) << "%" << endl;

            cout << endl;
            
            for (int i = 0; i < data.size(); ++i) {
                cout << setfill(' ') << setw(3) << (i + 1) << ": ";
                cout << "Fold: " << setfill(' ') << setw(2) << foldIdx[i];
                cout << "  Actual: " << actualLabel[i] << "  Predicted: " << predictedLabel[i];
                cout << "  Confidence: " << setprecision(6) << fixed << predictedConfidence[i] << endl;
            }
        }
    }
}