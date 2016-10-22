#ifndef FTRL_PREDICTOR_H_
#define FTRL_PREDICTOR_H_

#include "../Frame/pc_frame.h"
#include "ftrl_model.h"
#include "../Sample/fm_sample.h"


class ftrl_predictor : public pc_task
{
public:
    ftrl_predictor(int _class_num, int _factor_num, ifstream& _fModel, ofstream& _fPredict);
    virtual void run_task(vector<string>& dataBuffer);
private:
	int class_num;
    ftrl_model* pModel;
    ofstream& fPredict;
    mutex outMtx;
};


ftrl_predictor::ftrl_predictor(int _class_num, int _factor_num, ifstream& _fModel, ofstream& _fPredict):fPredict(_fPredict)
{
	class_num = _class_num;
    pModel = new ftrl_model(_class_num, _factor_num);
    if(!pModel->loadModel(_fModel))
    {
        cout << "load model error!" << endl;
        exit(-1);
    }
}

void ftrl_predictor::run_task(vector<string>& dataBuffer)
{
    vector<string> outputVec(dataBuffer.size());
	vector<double> biasVec(class_num);
    for(int i = 0; i < dataBuffer.size(); ++i)
    {
        fm_sample sample(dataBuffer[i], class_num);
		for(int k = 0; k < class_num; ++k)
		{
			biasVec[k] = pModel->muBias->mcu[k].wi;
		}
        vector<double> scoreVec = pModel->getScore(sample.x, biasVec, pModel->muMap);
        outputVec[i] = to_string(sample.y);
		for(int k = 0; k < class_num; ++k)
		{
			outputVec[i] += " " + to_string(scoreVec[k]);
		}
    }
    outMtx.lock();
    for(int i = 0; i < outputVec.size(); ++i)
    {
        fPredict << outputVec[i] << endl;
    }
    outMtx.unlock();
}


#endif /*FTRL_PREDICTOR_H_*/
