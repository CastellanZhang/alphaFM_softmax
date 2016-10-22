#ifndef FM_SAMPLE_H_
#define FM_SAMPLE_H_

#include <string>
#include <vector>

using namespace std;

const string spliter = " ";
const string innerSpliter = ":";


class fm_sample
{
public:
    int y;
    vector<pair<string, double> > x;
    fm_sample(const string& line, int class_num);
};


fm_sample::fm_sample(const string& line, int class_num)
{
    this->x.clear();
    size_t posb = line.find_first_not_of(spliter, 0);
    size_t pose = line.find_first_of(spliter, posb);
    int label = atoi(line.substr(posb, pose-posb).c_str());
	if(label < 1 || label > class_num)
	{
		cout << "wrong line input, label out of range\n" << line << endl;
        throw "wrong line input";
	}
    this->y = label;
    string key;
    double value;
    while(pose < line.size())
    {
        posb = line.find_first_not_of(spliter, pose);
        if(posb == string::npos)
        {
            break;
        }
        pose = line.find_first_of(innerSpliter, posb);
        if(pose == string::npos)
        {
            cout << "wrong line input\n" << line << endl;
            throw "wrong line input";
        }
        key = line.substr(posb, pose-posb);
        posb = pose + 1;
        if(posb >= line.size())
        {
            cout << "wrong line input\n" << line << endl;
            throw "wrong line input";
        }
        pose = line.find_first_of(spliter, posb);
        value = stod(line.substr(posb, pose-posb));
        if(value != 0)
        {
            this->x.push_back(make_pair(key, value));
        }
    }
}


#endif /*FM_SAMPLE_H_*/
