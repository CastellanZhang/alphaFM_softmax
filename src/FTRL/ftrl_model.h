#ifndef FTRL_MODEL_H_
#define FTRL_MODEL_H_

#include <unordered_map>
#include <string>
#include <vector>
#include <mutex>
#include <iostream>
#include <cmath>
#include "../Utils/utils.h"

using namespace std;


//每一个特征维度的每一个类别的模型单元
class ftrl_model_class_unit
{
public:
    double wi;
    double w_ni;
    double w_zi;
    vector<double> vi;
    vector<double> v_ni;
    vector<double> v_zi;
public:
    ftrl_model_class_unit(int factor_num, double v_mean, double v_stdev)
    {
        wi = 0.0;
        w_ni = 0.0;
        w_zi = 0.0;
        vi.resize(factor_num);
        v_ni.resize(factor_num);
        v_zi.resize(factor_num);
        for(int f = 0; f < factor_num; ++f)
        {
            vi[f] = utils::gaussian(v_mean, v_stdev);
            v_ni[f] = 0.0;
            v_zi[f] = 0.0;
        }
    }

    ftrl_model_class_unit(int factor_num, const vector<string>& modelLineSeg, int start)
    {
        vi.resize(factor_num);
        v_ni.resize(factor_num);
        v_zi.resize(factor_num);
        wi = stod(modelLineSeg[start + 1]);
        w_ni = stod(modelLineSeg[start + 2 + factor_num]);
        w_zi = stod(modelLineSeg[start + 3 + factor_num]);
        for(int f = 0; f < factor_num; ++f)
        {
            vi[f] = stod(modelLineSeg[start + 2 + f]);
            v_ni[f] = stod(modelLineSeg[start + 4 + factor_num + f]);
            v_zi[f] = stod(modelLineSeg[start + 4 + 2 * factor_num + f]);
        }
    }

    friend inline ostream& operator <<(ostream& os, const ftrl_model_class_unit& mcu)
    {
        os << mcu.wi;
        for(int f = 0; f < mcu.vi.size(); ++f)
        {
            os << " " << mcu.vi[f];
        }
        os << " " << mcu.w_ni << " " << mcu.w_zi;
        for(int f = 0; f < mcu.v_ni.size(); ++f)
        {
            os << " " << mcu.v_ni[f];
        }
        for(int f = 0; f < mcu.v_zi.size(); ++f)
        {
            os << " " << mcu.v_zi[f];
        }
        return os;
    }
};



//每一个特征维度的模型单元
class ftrl_model_unit
{
public:
	int class_num;
	vector<ftrl_model_class_unit> mcu;
    mutex mtx;
public:
    ftrl_model_unit(int cn, int factor_num, double v_mean, double v_stdev)
    {
		class_num = cn;
		for(int i = 0; i < class_num; ++i)
		{
			mcu.push_back(ftrl_model_class_unit(factor_num, v_mean, v_stdev));
		}
    }

    ftrl_model_unit(int cn, int factor_num, const vector<string>& modelLineSeg)
    {
		class_num = cn;
		for(int i = 0; i < class_num; ++i)
		{
			int start = i * (3 + 3 * factor_num);
			mcu.push_back(ftrl_model_class_unit(factor_num, modelLineSeg, start));
		}
    }

    friend inline ostream& operator <<(ostream& os, const ftrl_model_unit& mu)
    {
		for(int i = 0; i < mu.class_num; ++i)
		{
			if(i > 0) os << " ";
			os << mu.mcu[i];
		}
		return os;
    }
};




class ftrl_model
{
public:
    ftrl_model_unit* muBias;
    unordered_map<string, ftrl_model_unit*> muMap;

	int class_num;
    int factor_num;
    double init_stdev;
    double init_mean;

public:
    ftrl_model(int _num_class, int _factor_num);
    ftrl_model(int _num_class, int _factor_num, double _mean, double _stdev);
    ftrl_model_unit* getOrInitModelUnit(string index);
    ftrl_model_unit* getOrInitModelUnitBias();

    vector<double> predict(const vector<pair<string, double> >& x, const vector<double>& bias, vector<ftrl_model_unit*>& theta, vector<vector<double> >& sum);
    vector<double> getScore(const vector<pair<string, double> >& x, const vector<double>& bias, unordered_map<string, ftrl_model_unit*>& theta);
    void outputModel(ofstream& out);
    bool loadModel(ifstream& in);
    void debugPrintModel();

private:
    double get_wi(unordered_map<string, ftrl_model_unit*>& theta, const string& index, int classIndex);
    double get_vif(unordered_map<string, ftrl_model_unit*>& theta, const string& index, int f, int classIndex);
private:
    mutex mtx;
    mutex mtx_bias;
};


ftrl_model::ftrl_model(int _class_num, int _factor_num)
{
	class_num = _class_num;
    factor_num = _factor_num;
    init_mean = 0.0;
    init_stdev = 0.0;
    muBias = NULL;
}

ftrl_model::ftrl_model(int _class_num, int _factor_num, double _mean, double _stdev)
{
	class_num = _class_num;
    factor_num = _factor_num;
    init_mean = _mean;
    init_stdev = _stdev;
    muBias = NULL;
}


ftrl_model_unit* ftrl_model::getOrInitModelUnit(string index)
{
    unordered_map<string, ftrl_model_unit*>::iterator iter = muMap.find(index);
    if(iter == muMap.end())
    {
        mtx.lock();
        ftrl_model_unit* pMU = new ftrl_model_unit(class_num, factor_num, init_mean, init_stdev);
        muMap.insert(make_pair(index, pMU));
        mtx.unlock();
        return pMU;
    }
    else
    {
        return iter->second;
    }
}


ftrl_model_unit* ftrl_model::getOrInitModelUnitBias()
{
    if(NULL == muBias)
    {
        mtx_bias.lock();
        muBias = new ftrl_model_unit(class_num, 0, init_mean, init_stdev);
        mtx_bias.unlock();
    }
    return muBias;
}


vector<double> ftrl_model::predict(const vector<pair<string, double> >& x, const vector<double>& bias, vector<ftrl_model_unit*>& theta, vector<vector<double> >& sum)
{
	vector<double> resVec(class_num);
	for(int k = 0; k < class_num; ++k)
	{
		double result = 0;
		result += bias[k];
		for(int i = 0; i < x.size(); ++i)
		{
			result += theta[i]->mcu[k].wi * x[i].second;
		}
		double sum_sqr, d;
		for(int f = 0; f < factor_num; ++f)
		{
			sum[k][f] = sum_sqr = 0.0;
			for(int i = 0; i < x.size(); ++i)
			{
				d = theta[i]->mcu[k].vi[f] * x[i].second;
				sum[k][f] += d;
				sum_sqr += d * d;
			}
			result += 0.5 * (sum[k][f] * sum[k][f] - sum_sqr);
		}
		resVec[k] = result;
	}
	return resVec;
}


vector<double> ftrl_model::getScore(const vector<pair<string, double> >& x, const vector<double>& bias, unordered_map<string, ftrl_model_unit*>& theta)
{
	vector<double> scoreVec(class_num);
	double denominator = 0.0;
    double maxResult = numeric_limits<double>::lowest();
	for(int k = 0; k < class_num; ++k)
	{
		double result = 0;
		result += bias[k];
		for(int i = 0; i < x.size(); ++i)
		{
			result += get_wi(theta, x[i].first, k) * x[i].second;
		}
		double sum, sum_sqr, d;
		for(int f = 0; f < factor_num; ++f)
		{
			sum = sum_sqr = 0.0;
			for(int i = 0; i < x.size(); ++i)
			{
				d = get_vif(theta, x[i].first, f, k) * x[i].second;
				sum += d;
				sum_sqr += d * d;
			}
			result += 0.5 * (sum * sum - sum_sqr);
		}
        scoreVec[k] = result;
        if(result > maxResult) maxResult = result;
	}
    for(int k = 0; k < class_num; ++k)
    {
        scoreVec[k] -= maxResult;
        scoreVec[k] = exp(scoreVec[k]);
        denominator += scoreVec[k];
    }
	for(int k = 0; k < class_num; ++k)
	{
		scoreVec[k] /= denominator;
	}
	return scoreVec;
}


double ftrl_model::get_wi(unordered_map<string, ftrl_model_unit*>& theta, const string& index, int classIndex)
{
    unordered_map<string, ftrl_model_unit*>::iterator iter = theta.find(index);
    if(iter == theta.end())
    {
        return 0.0;
    }
    else
    {
        return iter->second->mcu[classIndex].wi;
    }
}


double ftrl_model::get_vif(unordered_map<string, ftrl_model_unit*>& theta, const string& index, int f, int classIndex)
{
    unordered_map<string, ftrl_model_unit*>::iterator iter = theta.find(index);
    if(iter == theta.end())
    {
        return 0.0;
    }
    else
    {
        return iter->second->mcu[classIndex].vi[f];
    }
}


void ftrl_model::outputModel(ofstream& out)
{
    out << "bias " << *muBias << endl;
    for(unordered_map<string, ftrl_model_unit*>::iterator iter = muMap.begin(); iter != muMap.end(); ++iter)
    {
        out << iter->first << " " << *(iter->second) << endl;
    }
}


void ftrl_model::debugPrintModel()
{
    cout << "bias " << *muBias << endl;
    for(unordered_map<string, ftrl_model_unit*>::iterator iter = muMap.begin(); iter != muMap.end(); ++iter)
    {
        cout << iter->first << " " << *(iter->second) << endl;
    }
}


bool ftrl_model::loadModel(ifstream& in)
{
    string line;
    if(!getline(in, line))
    {
        return false;
    }
    vector<string> strVec;
    utils::splitString(line, ' ', &strVec);
    if(strVec.size() != 1 + 3 * class_num)
    {
        return false;
    }
    muBias = new ftrl_model_unit(class_num, 0, strVec);
	const int SEG_NUM = 1 + (3 + 3 * factor_num) * class_num;
    while(getline(in, line))
    {
        strVec.clear();
        utils::splitString(line, ' ', &strVec);
        if(strVec.size() != SEG_NUM)
        {
            return false;
        }
        string& index = strVec[0];
        ftrl_model_unit* pMU = new ftrl_model_unit(class_num, factor_num, strVec);
        muMap[index] = pMU;
    }
    return true;
}



#endif /*FTRL_MODEL_H_*/
