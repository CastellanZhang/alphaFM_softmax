#ifndef FTRL_TRAINER_H_
#define FTRL_TRAINER_H_

#include "../Frame/pc_frame.h"
#include "ftrl_model.h"
#include "../Sample/fm_sample.h"
#include "../Utils/utils.h"


struct trainer_option
{
    trainer_option() : k0(true), k1(true), factor_num(8), init_mean(0.0), init_stdev(0.1), w_alpha(0.05), w_beta(1.0), w_l1(0.1), w_l2(5.0),
               v_alpha(0.05), v_beta(1.0), v_l1(0.1), v_l2(5.0), 
               threads_num(1), b_init(false), class_num(0) {}
    string model_path, init_m_path;
    double init_mean, init_stdev;
    double w_alpha, w_beta, w_l1, w_l2;
    double v_alpha, v_beta, v_l1, v_l2;
    int threads_num, factor_num, class_num;
    bool k0, k1, b_init;
    
    void parse_option(const vector<string>& args) 
    {
        int argc = args.size();
        if(0 == argc) throw invalid_argument("invalid command\n");
        for(int i = 0; i < argc; ++i)
        {
            if(args[i].compare("-m") == 0) 
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                model_path = args[++i];
            }
            else if(args[i].compare("-dim") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                vector<string> strVec;
                string tmpStr = args[++i];
                utils::splitString(tmpStr, ',', &strVec);
                if(strVec.size() != 3)
                    throw invalid_argument("invalid command\n");
                k0 = 0 == stoi(strVec[0]) ? false : true;
                k1 = 0 == stoi(strVec[1]) ? false : true;
                factor_num = stoi(strVec[2]);
            }
            else if(args[i].compare("-init_stdev") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                init_stdev = stod(args[++i]);
            }
            else if(args[i].compare("-w_alpha") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                w_alpha = stod(args[++i]);
            }
            else if(args[i].compare("-w_beta") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                w_beta = stod(args[++i]);
            }
            else if(args[i].compare("-w_l1") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                w_l1 = stod(args[++i]);
            }
            else if(args[i].compare("-w_l2") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                w_l2 = stod(args[++i]);
            }
            else if(args[i].compare("-v_alpha") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                v_alpha = stod(args[++i]);
            }
            else if(args[i].compare("-v_beta") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                v_beta = stod(args[++i]);
            }
            else if(args[i].compare("-v_l1") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                v_l1 = stod(args[++i]);
            }
            else if(args[i].compare("-v_l2") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                v_l2 = stod(args[++i]);
            }
            else if(args[i].compare("-core") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                threads_num = stoi(args[++i]);
            }
            else if(args[i].compare("-im") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                init_m_path = args[++i];
                b_init = true; //if im field exits , that means b_init = true !
            }
			else if(args[i].compare("-cn") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                class_num = stoi(args[++i]);
            }
            else
            {
                throw invalid_argument("invalid command\n");
                break;
            }
        }
    }

};


class ftrl_trainer : public pc_task
{
public:
    ftrl_trainer(const trainer_option& opt);
    virtual void run_task(vector<string>& dataBuffer);
    bool loadModel(ifstream& in);
    void outputModel(ofstream& out);
private:
    void train(int y, const vector<pair<string, double> >& x);
private:
    ftrl_model* pModel;
    double w_alpha, w_beta, w_l1, w_l2;
    double v_alpha, v_beta, v_l1, v_l2;
    bool k0;
    bool k1;
};


ftrl_trainer::ftrl_trainer(const trainer_option& opt)
{
    w_alpha = opt.w_alpha;
    w_beta = opt.w_beta;
    w_l1 = opt.w_l1;
    w_l2 = opt.w_l2;
    v_alpha = opt.v_alpha;
    v_beta = opt.v_beta;
    v_l1 = opt.v_l1;
    v_l2 = opt.v_l2;
    k0 = opt.k0;
    k1 = opt.k1;
    pModel = new ftrl_model(opt.class_num, opt.factor_num, opt.init_mean, opt.init_stdev);
}

void ftrl_trainer::run_task(vector<string>& dataBuffer)
{
	int class_num = pModel->class_num;
    for(int i = 0; i < dataBuffer.size(); ++i)
    {
        fm_sample sample(dataBuffer[i], class_num);
        train(sample.y, sample.x);
    }
}


bool ftrl_trainer::loadModel(ifstream& in)
{
    return pModel->loadModel(in);
}


void ftrl_trainer::outputModel(ofstream& out)
{
    return pModel->outputModel(out);
}


//输入一个样本，更新参数
void ftrl_trainer::train(int y, const vector<pair<string, double> >& x)
{
    ftrl_model_unit* thetaBias = pModel->getOrInitModelUnitBias();
    vector<ftrl_model_unit*> theta(x.size(), NULL);
    int xLen = x.size();
    for(int i = 0; i < xLen; ++i)
    {
        const string& index = x[i].first;
        theta[i] = pModel->getOrInitModelUnit(index);
    }
    //update w via FTRL
    for(int i = 0; i <= xLen; ++i)
    {
        ftrl_model_unit& mu = i < xLen ? *(theta[i]) : *thetaBias;
        if((i < xLen && k1) || (i == xLen && k0))
        {
            mu.mtx.lock();
			for(int k = 0; k < pModel->class_num; ++k)
			{
				if(fabs(mu.mcu[k].w_zi) <= w_l1)
				{
					mu.mcu[k].wi = 0.0;
				}
				else
				{
					mu.mcu[k].wi = (-1) *
						(1 / (w_l2 + (w_beta + sqrt(mu.mcu[k].w_ni)) / w_alpha)) *
						(mu.mcu[k].w_zi - utils::sgn(mu.mcu[k].w_zi) * w_l1);
				}
			}
            mu.mtx.unlock();
        }
    }
    //update v via FTRL
    for(int i = 0; i < xLen; ++i)
    {
        ftrl_model_unit& mu = *(theta[i]);
        for(int f = 0; f < pModel->factor_num; ++f)
        {
            mu.mtx.lock();
			for(int k = 0; k < pModel->class_num; ++k)
			{
				double& vif = mu.mcu[k].vi[f];
				double& v_nif = mu.mcu[k].v_ni[f];
				double& v_zif = mu.mcu[k].v_zi[f];
				if(v_nif > 0)
				{
					if(fabs(v_zif) <= v_l1)
					{
						vif = 0.0;
					}
					else
					{
						vif = (-1) *
							(1 / (v_l2 + (v_beta + sqrt(v_nif)) / v_alpha)) *
							(v_zif - utils::sgn(v_zif) * v_l1);
					}
				}
			}
            mu.mtx.unlock();
        }
    }
	vector<vector<double> > sum(pModel->class_num);
	vector<double> biasVec(pModel->class_num);
	for(int k = 0; k < pModel->class_num; ++k)
	{
		sum[k].resize(pModel->factor_num);
		biasVec[k] = thetaBias->mcu[k].wi;
	}
	vector<double> p = pModel->predict(x, biasVec, theta, sum);
    double max_p = numeric_limits<double>::lowest();
    for(int k = 0; k < pModel->class_num; ++k)
    {
        if(p[k] > max_p) max_p = p[k];
    }
    double denominator = 0.0;
    for(int k = 0; k < pModel->class_num; ++k)
    {
        p[k] -= max_p;
        p[k] = exp(p[k]);
        denominator += p[k];
    }
    for(int k = 0; k < pModel->class_num; ++k)
    {
        p[k] /= denominator;
    }
	vector<double> mult(pModel->class_num);
	for(int k = 0; k < pModel->class_num; ++k)
	{
		int yk = y == (k + 1) ? 1 : 0;
		mult[k] = p[k] - yk;
	}
    //update w_n, w_z
    for(int i = 0; i <= xLen; ++i)
    {
        ftrl_model_unit& mu = i < xLen ? *(theta[i]) : *thetaBias;
        double xi = i < xLen ? x[i].second : 1.0;
        if((i < xLen && k1) || (i == xLen && k0))
        {
            mu.mtx.lock();
			for(int k = 0; k < pModel->class_num; ++k)
			{
				double w_gi = mult[k] * xi;
				double w_si = 1 / w_alpha * (sqrt(mu.mcu[k].w_ni + w_gi * w_gi) - sqrt(mu.mcu[k].w_ni));
				mu.mcu[k].w_zi += w_gi - w_si * mu.mcu[k].wi;
				mu.mcu[k].w_ni += w_gi * w_gi;
			}
            mu.mtx.unlock();
        }
    }
    //update v_n, v_z
    for(int i = 0; i < xLen; ++i)
    {
        ftrl_model_unit& mu = *(theta[i]);
        const double& xi = x[i].second;
        for(int f = 0; f < pModel->factor_num; ++f)
        {
            mu.mtx.lock();
			for(int k = 0; k < pModel->class_num; ++k)
			{
				double& vif = mu.mcu[k].vi[f];
				double& v_nif = mu.mcu[k].v_ni[f];
				double& v_zif = mu.mcu[k].v_zi[f];
				double v_gif = mult[k] * (sum[k][f] * xi - vif * xi * xi);
				double v_sif = 1 / v_alpha * (sqrt(v_nif + v_gif * v_gif) - sqrt(v_nif));
				v_zif += v_gif - v_sif * vif;
				v_nif += v_gif * v_gif;
			}
            mu.mtx.unlock();
        }
    }
    //////////
    //pModel->debugPrintModel();
    //////////
}


#endif /*FTRL_TRAINER_H_*/
